//! Minimal NIfTI-1 reader for GCLDA masks.
//!
//! Only the fields needed to build [`MaskInfo`] are parsed: this is not a
//! general-purpose NIfTI library. Both gzip-compressed (`.nii.gz`) and plain
//! (`.nii`) single files are supported. Affine selection matches nibabel's
//! `get_best_affine` (sform, then qform, then a `pixdim` diagonal).
//!
//! Voxel data on disk is stored in Fortran order (the `i`/x axis varies
//! fastest), but `np.where` on the in-memory array walks it in C order (`i`
//! slowest, `k` fastest). We iterate output rows in that C order while
//! computing the on-disk element index with Fortran strides.

use crate::GcldaError;
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Nonzero-voxel coordinates and geometry for a NIfTI mask image.
pub struct MaskInfo {
    /// One row per nonzero voxel, in the C order that `np.where` produces
    /// (`i` slowest, `k` fastest).
    pub xyz: Vec<[f64; 3]>,
    pub affine: [[f64; 4]; 4],
    pub shape: [usize; 3],
}

fn read_i16(buf: &[u8], off: usize, big_endian: bool) -> i16 {
    let b: [u8; 2] = buf[off..off + 2].try_into().unwrap();
    if big_endian {
        i16::from_be_bytes(b)
    } else {
        i16::from_le_bytes(b)
    }
}

fn read_i32(buf: &[u8], off: usize, big_endian: bool) -> i32 {
    let b: [u8; 4] = buf[off..off + 4].try_into().unwrap();
    if big_endian {
        i32::from_be_bytes(b)
    } else {
        i32::from_le_bytes(b)
    }
}

fn read_f32(buf: &[u8], off: usize, big_endian: bool) -> f32 {
    let b: [u8; 4] = buf[off..off + 4].try_into().unwrap();
    if big_endian {
        f32::from_be_bytes(b)
    } else {
        f32::from_le_bytes(b)
    }
}

fn read_f64(buf: &[u8], off: usize, big_endian: bool) -> f64 {
    let b: [u8; 8] = buf[off..off + 8].try_into().unwrap();
    if big_endian {
        f64::from_be_bytes(b)
    } else {
        f64::from_le_bytes(b)
    }
}

/// Build a 3x3 rotation matrix from a unit quaternion (w, x, y, z), matching
/// nibabel's `nibabel.quaternions.quat2mat`.
fn quat_to_mat(w: f64, x: f64, y: f64, z: f64) -> [[f64; 3]; 3] {
    let nq = w * w + x * x + y * y + z * z;
    if nq < 1e-14 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let s = 2.0 / nq;
    let (xs, ys, zs) = (x * s, y * s, z * s);
    let (wx, wy, wz) = (w * xs, w * ys, w * zs);
    let (xx, xy, xz) = (x * xs, x * ys, x * zs);
    let (yy, yz, zz) = (y * ys, y * zs, z * zs);
    [
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ]
}

/// Read the raw bytes of a NIfTI-1 single file, transparently decompressing
/// gzip (detected by the `1f 8b` magic) rather than relying on the file
/// extension.
fn read_all_bytes(path: &Path) -> Result<Vec<u8>, GcldaError> {
    let mut probe = [0u8; 2];
    let n = {
        let mut file = File::open(path)?;
        file.read(&mut probe)?
    };
    let mut out = Vec::new();
    if n == 2 && probe == [0x1f, 0x8b] {
        let file = File::open(path)?;
        let mut decoder = GzDecoder::new(file);
        decoder.read_to_end(&mut out)?;
    } else {
        let mut file = File::open(path)?;
        file.read_to_end(&mut out)?;
    }
    Ok(out)
}

/// Load a NIfTI-1 mask image and return the xyz coordinates of its nonzero
/// voxels, in the same order `np.where(_mask_img_to_bool(img))` followed by
/// `nib.affines.apply_affine` would produce.
pub fn load_mask_xyz(path: &Path) -> Result<MaskInfo, GcldaError> {
    let buf = read_all_bytes(path)?;
    if buf.len() < 352 {
        return Err(GcldaError::Parse("NIfTI file too short".to_string()));
    }

    let sizeof_hdr_le = i32::from_le_bytes(buf[0..4].try_into().unwrap());
    let big_endian = if sizeof_hdr_le == 348 {
        false
    } else if sizeof_hdr_le == 1_543_569_408 {
        true
    } else {
        return Err(GcldaError::Parse(format!(
            "not a NIfTI-1 file (sizeof_hdr = {sizeof_hdr_le})"
        )));
    };

    let dim = |i: usize| read_i16(&buf, 40 + 2 * i, big_endian) as usize;
    let nx = dim(1);
    let ny = dim(2);
    let nz = dim(3);

    let datatype = read_i16(&buf, 70, big_endian);
    let bitpix = read_i16(&buf, 72, big_endian);
    let pixdim = |i: usize| read_f32(&buf, 76 + 4 * i, big_endian) as f64;

    let vox_offset = read_f32(&buf, 108, big_endian) as f64;
    // scl_slope/scl_inter are parsed for header completeness but deliberately
    // unused: the nonzero test operates on raw stored values, matching
    // nibabel's `astype(bool)` on `dataobj` (no scaling applied).
    let _scl_slope = read_f32(&buf, 112, big_endian);
    let _scl_inter = read_f32(&buf, 116, big_endian);

    let qform_code = read_i16(&buf, 252, big_endian);
    let sform_code = read_i16(&buf, 254, big_endian);

    let affine: [[f64; 4]; 4] = if sform_code != 0 {
        let row = |base: usize| -> [f64; 4] {
            [
                read_f32(&buf, base, big_endian) as f64,
                read_f32(&buf, base + 4, big_endian) as f64,
                read_f32(&buf, base + 8, big_endian) as f64,
                read_f32(&buf, base + 12, big_endian) as f64,
            ]
        };
        [row(280), row(296), row(312), [0.0, 0.0, 0.0, 1.0]]
    } else if qform_code != 0 {
        let b = read_f32(&buf, 256, big_endian) as f64;
        let c = read_f32(&buf, 260, big_endian) as f64;
        let d = read_f32(&buf, 264, big_endian) as f64;
        let sum_sq = b * b + c * c + d * d;
        let w = (1.0 - sum_sq).max(0.0).sqrt();
        let r = quat_to_mat(w, b, c, d);

        let qfac_raw = pixdim(0);
        let qfac = if qfac_raw == 1.0 || qfac_raw == -1.0 {
            qfac_raw
        } else {
            1.0
        };
        let vox = [pixdim(1), pixdim(2), pixdim(3) * qfac];

        let qoffset_x = read_f32(&buf, 268, big_endian) as f64;
        let qoffset_y = read_f32(&buf, 272, big_endian) as f64;
        let qoffset_z = read_f32(&buf, 276, big_endian) as f64;

        let mut m = [[0.0; 4]; 4];
        for row in 0..3 {
            for (col, vox_col) in vox.iter().enumerate() {
                m[row][col] = r[row][col] * vox_col;
            }
        }
        m[0][3] = qoffset_x;
        m[1][3] = qoffset_y;
        m[2][3] = qoffset_z;
        m[3] = [0.0, 0.0, 0.0, 1.0];
        m
    } else {
        [
            [pixdim(1), 0.0, 0.0, 0.0],
            [0.0, pixdim(2), 0.0, 0.0],
            [0.0, 0.0, pixdim(3), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    };

    let data_start = if vox_offset == 0.0 {
        352
    } else {
        vox_offset as usize
    };
    let bytes_per_voxel = (bitpix / 8) as usize;
    let n_voxels_total = nx * ny * nz;
    let needed = data_start + n_voxels_total * bytes_per_voxel;
    if buf.len() < needed {
        return Err(GcldaError::Parse(format!(
            "NIfTI data truncated: need {needed} bytes, have {}",
            buf.len()
        )));
    }

    // On disk, voxel data is stored in Fortran order: i (x) varies fastest,
    // then j (y), then k (z) slowest. This is independent of the C-order
    // iteration below, which matches np.where's traversal of the logical
    // (nx, ny, nz) array.
    let raw_index = |i: usize, j: usize, k: usize| -> usize { i + j * nx + k * nx * ny };

    let is_nonzero = |i: usize, j: usize, k: usize| -> Result<bool, GcldaError> {
        let off = data_start + raw_index(i, j, k) * bytes_per_voxel;
        let v = &buf[off..off + bytes_per_voxel];
        let nonzero = match datatype {
            2 => v[0] != 0,                             // uint8
            4 => read_i16(v, 0, big_endian) != 0,        // int16
            8 => read_i32(v, 0, big_endian) != 0,        // int32
            16 => read_f32(v, 0, big_endian) != 0.0,     // float32
            64 => read_f64(v, 0, big_endian) != 0.0,     // float64
            other => {
                return Err(GcldaError::Parse(format!(
                    "unsupported NIfTI datatype {other}"
                )))
            }
        };
        Ok(nonzero)
    };

    let mut xyz = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if is_nonzero(i, j, k)? {
                    let (fi, fj, fk) = (i as f64, j as f64, k as f64);
                    let x = affine[0][0] * fi + affine[0][1] * fj + affine[0][2] * fk + affine[0][3];
                    let y = affine[1][0] * fi + affine[1][1] * fj + affine[1][2] * fk + affine[1][3];
                    let z = affine[2][0] * fi + affine[2][1] * fj + affine[2][2] * fk + affine[2][3];
                    xyz.push([x, y, z]);
                }
            }
        }
    }

    Ok(MaskInfo {
        xyz,
        affine,
        shape: [nx, ny, nz],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal, valid little-endian NIfTI-1 single-file `.nii` image
    /// by hand: a 2x2x2 uint8 volume, an identity-plus-translation sform
    /// affine, and `vox_offset == 0.0` (so data must start at byte 352, the
    /// 348-byte header plus the 4-byte extension field). Bytes 348..352 (the
    /// extension field) are set to a nonzero canary that is not valid voxel
    /// data: if the reader mistakenly starts reading 4 bytes early (at 348
    /// instead of 352), it picks up the canary as the first voxels and gets
    /// the wrong nonzero set entirely.
    ///
    /// This exists because the bundled MNI mask used in `mask_golden.rs`
    /// turns out not to exercise this fallback at all: it carries an AFNI
    /// header extension, so its true `vox_offset` is 448, not 0 (see the
    /// task 7 follow-up report). That real file only ever exercises the
    /// `vox_offset != 0` branch, so this synthetic file is the only thing in
    /// the suite that proves the `352` fallback offset is right.
    fn minimal_nifti_bytes() -> Vec<u8> {
        let mut buf = vec![0u8; 352 + 8];

        buf[0..4].copy_from_slice(&348i32.to_le_bytes());

        // dim[0..8] at offset 40: a 3D, 2x2x2 volume.
        let dim: [i16; 8] = [3, 2, 2, 2, 1, 1, 1, 1];
        for (i, d) in dim.iter().enumerate() {
            buf[40 + 2 * i..42 + 2 * i].copy_from_slice(&d.to_le_bytes());
        }

        buf[70..72].copy_from_slice(&2i16.to_le_bytes()); // datatype: uint8
        buf[72..74].copy_from_slice(&8i16.to_le_bytes()); // bitpix

        buf[108..112].copy_from_slice(&0.0f32.to_le_bytes()); // vox_offset

        buf[252..254].copy_from_slice(&0i16.to_le_bytes()); // qform_code
        buf[254..256].copy_from_slice(&1i16.to_le_bytes()); // sform_code

        // sform rows: identity rotation/scale, translation (10, 20, 30).
        let srow_x: [f32; 4] = [1.0, 0.0, 0.0, 10.0];
        let srow_y: [f32; 4] = [0.0, 1.0, 0.0, 20.0];
        let srow_z: [f32; 4] = [0.0, 0.0, 1.0, 30.0];
        for (base, row) in [(280usize, srow_x), (296, srow_y), (312, srow_z)] {
            for (i, v) in row.iter().enumerate() {
                buf[base + 4 * i..base + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
            }
        }

        buf[344..348].copy_from_slice(b"n+1\0"); // magic

        // Extension field: a nonzero canary, not a valid "no extension"
        // marker. The reader must ignore its content and still start voxel
        // data at 352.
        buf[348..352].copy_from_slice(&[9, 9, 9, 9]);

        // Voxel data at offset 352..360, Fortran order (i fastest): only
        // ijk = (0, 0, 1) -> linear index 0 + 0*2 + 1*4 = 4 -> nonzero.
        let voxels: [u8; 8] = [0, 0, 0, 0, 5, 0, 0, 0];
        buf[352..360].copy_from_slice(&voxels);

        buf
    }

    #[test]
    fn vox_offset_zero_reads_data_from_352_not_348() {
        let dir = std::env::temp_dir().join("gclda_nifti_unit_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synthetic.nii");
        std::fs::write(&path, minimal_nifti_bytes()).unwrap();

        let info = load_mask_xyz(&path).unwrap();

        assert_eq!(info.shape, [2, 2, 2]);
        assert_eq!(
            info.xyz.len(),
            1,
            "expected exactly one nonzero voxel, got {:?}",
            info.xyz
        );
        assert_eq!(info.xyz[0], [10.0, 20.0, 31.0]);
    }
}
