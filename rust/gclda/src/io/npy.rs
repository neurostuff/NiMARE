//! Minimal NPY v1.0 writer.
//!
//! The format is a short ASCII header followed by raw little-endian data,
//! which lets Python open large outputs with np.load(..., mmap_mode="r")
//! without ever making them resident.

use crate::GcldaError;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F64,
    F32,
    I64,
}

impl Dtype {
    fn descr(self) -> &'static str {
        match self {
            Dtype::F64 => "<f8",
            Dtype::F32 => "<f4",
            Dtype::I64 => "<i8",
        }
    }
}

fn header_bytes(shape: &[usize], dtype: Dtype) -> Vec<u8> {
    // NumPy writes a trailing comma for 1-D shapes: (5,) not (5)
    let shape_repr = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("({})", parts.join(", "))
    };
    let dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        dtype.descr(),
        shape_repr
    );

    // Magic (6) + version (2) + header length (2) + dict must be a multiple of 64.
    let mut padded = dict.into_bytes();
    let prefix = 10;
    let unpadded = prefix + padded.len() + 1; // +1 for the trailing newline
    let pad = (64 - (unpadded % 64)) % 64;
    padded.extend(std::iter::repeat(b' ').take(pad));
    padded.push(b'\n');

    let mut out = Vec::with_capacity(prefix + padded.len());
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1); // major
    out.push(0); // minor
    out.extend_from_slice(&(padded.len() as u16).to_le_bytes());
    out.extend_from_slice(&padded);
    out
}

pub struct NpyWriter {
    inner: BufWriter<File>,
    dtype: Dtype,
    expected: usize,
    written: usize,
}

impl NpyWriter {
    pub fn create(path: &Path, shape: &[usize], dtype: Dtype) -> Result<Self, GcldaError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut inner = BufWriter::new(File::create(path)?);
        inner.write_all(&header_bytes(shape, dtype))?;
        Ok(NpyWriter {
            inner,
            dtype,
            expected: shape.iter().product(),
            written: 0,
        })
    }

    pub fn write_row(&mut self, row: &[f64]) -> Result<(), GcldaError> {
        for &v in row {
            match self.dtype {
                Dtype::F64 => self.inner.write_all(&v.to_le_bytes())?,
                Dtype::F32 => self.inner.write_all(&(v as f32).to_le_bytes())?,
                Dtype::I64 => self.inner.write_all(&(v as i64).to_le_bytes())?,
            }
        }
        self.written += row.len();
        Ok(())
    }

    pub fn write_row_i64(&mut self, row: &[i64]) -> Result<(), GcldaError> {
        for &v in row {
            self.inner.write_all(&v.to_le_bytes())?;
        }
        self.written += row.len();
        Ok(())
    }

    pub fn finish(mut self) -> Result<(), GcldaError> {
        assert_eq!(
            self.written, self.expected,
            "npy writer got {} values, header declared {}",
            self.written, self.expected
        );
        self.inner.flush()?;
        Ok(())
    }
}

pub fn write_f64(path: &Path, shape: &[usize], data: &[f64]) -> Result<(), GcldaError> {
    let mut w = NpyWriter::create(path, shape, Dtype::F64)?;
    w.write_row(data)?;
    w.finish()
}

pub fn write_f32_from_f64(path: &Path, shape: &[usize], data: &[f64]) -> Result<(), GcldaError> {
    let mut w = NpyWriter::create(path, shape, Dtype::F32)?;
    w.write_row(data)?;
    w.finish()
}

pub fn write_i64(path: &Path, shape: &[usize], data: &[i64]) -> Result<(), GcldaError> {
    let mut w = NpyWriter::create(path, shape, Dtype::I64)?;
    w.write_row_i64(data)?;
    w.finish()
}
