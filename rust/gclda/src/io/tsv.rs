//! Streaming TSV ingest.
//!
//! Reads counts and coordinates directly into token-level index arrays. The
//! dense D x W count matrix that the Python constructor materializes (~340 MB
//! at Neurosynth scale) is never built.
//!
//! The index semantics here are load-bearing; see the task notes. In
//! particular, IDs sort as STRINGS ("10" < "2" < "9"), and count rows are
//! traversed in FILE order, not docidx order.

use crate::GcldaError;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct Corpus {
    pub ids: Vec<String>,
    pub vocabulary: Vec<String>,
    pub wtoken_doc_idx: Vec<u32>,
    pub wtoken_word_idx: Vec<u32>,
    pub ptoken_doc_idx: Vec<u32>,
    pub ptoken_coords: Vec<[f64; 3]>,
}

fn open(path: &Path) -> Result<BufReader<File>, GcldaError> {
    Ok(BufReader::new(File::open(path)?))
}

/// Split a TSV line into fields, stripping a trailing `\r` (CRLF files).
fn split_line(line: &str) -> Vec<&str> {
    line.strip_suffix('\r').unwrap_or(line).split('\t').collect()
}

fn parse_count(field: &str, context: &str) -> Result<i64, GcldaError> {
    if let Ok(v) = field.parse::<i64>() {
        return Ok(v);
    }
    // Fall back for counts written in float form (e.g. "2.0", "2.9"). NumPy's
    // float->int64 cast (`count_df[...].to_numpy(dtype=np.int64)`) truncates
    // toward zero, not rounds -- `as i64` on an f64 in Rust does the same, so
    // this must NOT be `.round()`: "2.9" must become 2, matching Python, not 3.
    //
    // Non-finite values must be rejected explicitly. Rust's `as i64` cast is
    // *saturating*: NaN -> 0 (a silent, wrong count) and +/-Inf -> i64::{MAX,MIN}
    // (which then makes the token-expansion `for _ in 0..count` loop attempt
    // ~9.2e18 iterations, i.e. an effective hang). Do NOT drop this guard to
    // "match" pandas: pandas' `to_numpy(dtype=np.int64)` cast is *also*
    // saturating (NaN and Inf both become i64::MIN, with a RuntimeWarning) --
    // but that value is negative, so it fails a few lines later in the Python
    // constructor at `np.repeat(..., counts)` with "repeats may not contain
    // negative values". Python therefore refuses this input too, just via a
    // confusing downstream error; we refuse it here, clearly and up front.
    field
        .parse::<f64>()
        .map_err(|e| GcldaError::Parse(format!("bad count {field:?} ({context}): {e}")))
        .and_then(|v| {
            if v.is_finite() {
                Ok(v as i64)
            } else {
                Err(GcldaError::Parse(format!("non-finite count {field:?} ({context})")))
            }
        })
}

fn parse_coord(field: &str, context: &str) -> Result<f64, GcldaError> {
    field
        .parse::<f64>()
        .map_err(|e| GcldaError::Parse(format!("bad coordinate {field:?} ({context}): {e}")))
}

fn column_index(header: &[&str], name: &str, context: &str) -> Result<usize, GcldaError> {
    header
        .iter()
        .position(|&h| h == name)
        .ok_or_else(|| GcldaError::Parse(format!("missing column {name:?} in {context}")))
}

pub fn load_corpus(counts: &Path, coords: &Path) -> Result<Corpus, GcldaError> {
    // --- Pass 1 over counts: header -> term names; first column -> count IDs. ---
    let mut reader = open(counts)?;
    let mut header_line = String::new();
    reader.read_line(&mut header_line)?;
    let header_fields = split_line(header_line.trim_end_matches('\n'));
    if header_fields.is_empty() {
        return Err(GcldaError::Parse("counts file has no header".to_string()));
    }
    let term_names: Vec<String> = header_fields[1..].iter().map(|s| s.to_string()).collect();
    let n_terms = term_names.len();

    let mut count_ids: HashSet<String> = HashSet::new();
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let fields = split_line(&line);
        let id = fields
            .first()
            .ok_or_else(|| GcldaError::Parse("counts row missing id column".to_string()))?;
        count_ids.insert((*id).to_string());
    }

    // --- Pass 1 over coordinates: collect coordinate IDs. ---
    let mut reader = open(coords)?;
    let mut header_line = String::new();
    reader.read_line(&mut header_line)?;
    let coord_header = split_line(header_line.trim_end_matches('\n'));
    let id_col = column_index(&coord_header, "id", "coordinates header")?;

    let mut coord_ids: HashSet<String> = HashSet::new();
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let fields = split_line(&line);
        let id = fields.get(id_col).ok_or_else(|| {
            GcldaError::Parse("coordinates row missing id column".to_string())
        })?;
        coord_ids.insert((*id).to_string());
    }

    // ids = sorted intersection of count IDs and coordinate IDs, compared as strings.
    let mut ids: Vec<String> = count_ids.intersection(&coord_ids).cloned().collect();
    ids.sort();
    let docidx_of: HashMap<&str, u32> =
        ids.iter().enumerate().map(|(i, id)| (id.as_str(), i as u32)).collect();

    // --- Pass 2 over counts: for retained rows, in file order, collect nonzero
    // (docidx, orig_col, count) triples and mark which columns are ever nonzero. ---
    let mut reader = open(counts)?;
    let mut discard = String::new();
    reader.read_line(&mut discard)?; // skip header

    let mut col_seen = vec![false; n_terms];
    // (docidx, orig_col, count) in row-file order, ascending column within row.
    let mut triples: Vec<(u32, u32, i64)> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let fields = split_line(&line);
        let id = fields
            .first()
            .ok_or_else(|| GcldaError::Parse("counts row missing id column".to_string()))?;
        let Some(&docidx) = docidx_of.get(id) else {
            continue; // not in the retained ID set
        };
        if fields.len() - 1 != n_terms {
            return Err(GcldaError::Parse(format!(
                "counts row for id {id:?} has {} term fields, expected {n_terms}",
                fields.len() - 1
            )));
        }
        for (col, field) in fields[1..].iter().enumerate() {
            let count = parse_count(field, &format!("counts id={id}"))?;
            if count == 0 {
                continue;
            }
            if count < 0 {
                return Err(GcldaError::Parse(format!(
                    "negative count {count} for id {id:?}, column {}",
                    term_names[col]
                )));
            }
            col_seen[col] = true;
            triples.push((docidx, col as u32, count));
        }
    }

    // vocabulary = term names whose column is nonzero in at least one retained
    // row, preserving original column order. Columns that are zero across all
    // retained documents are dropped in place.
    let mut vocabulary: Vec<String> = Vec::new();
    let mut new_col_of: Vec<Option<u32>> = vec![None; n_terms];
    for (col, name) in term_names.iter().enumerate() {
        if col_seen[col] {
            new_col_of[col] = Some(vocabulary.len() as u32);
            vocabulary.push(name.clone());
        }
    }

    // Expand tokens: rows already in file order, ascending column within row
    // (triples were pushed in that order above) -- this reproduces np.nonzero's
    // row-major traversal over the D x W matrix without ever materializing it.
    let mut wtoken_doc_idx: Vec<u32> = Vec::new();
    let mut wtoken_word_idx: Vec<u32> = Vec::new();
    for (docidx, orig_col, count) in triples {
        let new_col = new_col_of[orig_col as usize]
            .expect("nonzero triple must reference a surviving column");
        for _ in 0..count {
            wtoken_doc_idx.push(docidx);
            wtoken_word_idx.push(new_col);
        }
    }

    // --- Pass 2 over coordinates: retained rows, in file order. ---
    let mut reader = open(coords)?;
    let mut header_line = String::new();
    reader.read_line(&mut header_line)?;
    let coord_header = split_line(header_line.trim_end_matches('\n'));
    let id_col = column_index(&coord_header, "id", "coordinates header")?;
    let x_col = column_index(&coord_header, "x", "coordinates header")?;
    let y_col = column_index(&coord_header, "y", "coordinates header")?;
    let z_col = column_index(&coord_header, "z", "coordinates header")?;

    let mut ptoken_doc_idx: Vec<u32> = Vec::new();
    let mut ptoken_coords: Vec<[f64; 3]> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let fields = split_line(&line);
        let id = fields.get(id_col).ok_or_else(|| {
            GcldaError::Parse("coordinates row missing id column".to_string())
        })?;
        let Some(&docidx) = docidx_of.get(id) else {
            continue; // not in the retained ID set
        };
        let context = format!("coordinates id={id}");
        let x = parse_coord(fields.get(x_col).unwrap_or(&""), &context)?;
        let y = parse_coord(fields.get(y_col).unwrap_or(&""), &context)?;
        let z = parse_coord(fields.get(z_col).unwrap_or(&""), &context)?;
        ptoken_doc_idx.push(docidx);
        ptoken_coords.push([x, y, z]);
    }

    Ok(Corpus {
        ids,
        vocabulary,
        wtoken_doc_idx,
        wtoken_word_idx,
        ptoken_doc_idx,
        ptoken_coords,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NumPy's float->int64 cast (`count_df[...].to_numpy(dtype=np.int64)`)
    /// truncates toward zero, not rounds. A fractional count like "2.9" must
    /// parse to 2, matching Python, not 3 (which `.round()` would give).
    #[test]
    fn float_form_counts_truncate_toward_zero_like_numpy() {
        assert_eq!(parse_count("2.9", "test").unwrap(), 2);
        assert_eq!(parse_count("2.4", "test").unwrap(), 2);
        assert_eq!(parse_count("2.0", "test").unwrap(), 2);
        assert_eq!(parse_count("-1.7", "test").unwrap(), -1);
        assert_eq!(parse_count("3", "test").unwrap(), 3);
    }

    /// Rust's `as i64` cast on a float is *saturating*, not error-raising:
    /// NaN silently becomes 0 and +/-Inf silently become i64::{MAX,MIN}. Both
    /// are wrong (a silently different model, or an ~9.2e18-iteration hang in
    /// token expansion), so `parse_count` must reject non-finite input
    /// explicitly rather than relying on the cast.
    #[test]
    fn non_finite_counts_are_rejected() {
        assert!(parse_count("nan", "test").is_err());
        assert!(parse_count("inf", "test").is_err());
        assert!(parse_count("-inf", "test").is_err());
    }
}
