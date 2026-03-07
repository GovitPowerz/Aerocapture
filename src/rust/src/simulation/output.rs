//! Output file writers.
//!
//! Matches Fortran sortie.f, photra.f, result.f output formats.
//! Writes fort.201-204, photo.*, final.*, initial.* files.

use std::fs::File;
use std::io::{self, BufWriter, Write};

/// Fortran-compatible D-notation float formatter.
///
/// Formats a f64 as Fortran D-notation: " 0.12345678901234D+02"
pub fn fortran_float(val: f64, width: usize, decimals: usize) -> String {
    if val == 0.0 {
        let zeros: String = "0".repeat(decimals);
        return format!("{:>width$}", format!("0.{}D+00", zeros), width = width);
    }

    let sign = if val < 0.0 { "-" } else { " " };
    let abs_val = val.abs();
    let exp = abs_val.log10().floor() as i32;
    let mantissa = abs_val / 10.0_f64.powi(exp);

    // Adjust so mantissa is in [0.1, 1.0)
    let (mantissa, exp) = if mantissa >= 1.0 {
        (mantissa / 10.0, exp + 1)
    } else {
        (mantissa, exp)
    };

    let exp_sign = if exp >= 0 { "+" } else { "-" };
    let exp_abs = exp.unsigned_abs();

    let mant_str = format!("{:.prec$}", mantissa, prec = decimals);
    let full = format!("{}{}D{}{:02}", sign, mant_str, exp_sign, exp_abs);
    format!("{:>width$}", full, width = width)
}

/// Write a trajectory snapshot line (photo format).
///
/// Format: 24 columns, (24(1x,d12.5))
pub fn write_photo_line(writer: &mut impl Write, values: &[f64]) -> io::Result<()> {
    for val in values {
        write!(writer, " {}", fortran_float(*val, 12, 5))?;
    }
    writeln!(writer)?;
    Ok(())
}

/// Write a final conditions line (carltf.f format).
///
/// Format: i5 + 52 D15.7 values — matches `format(1x,i5,52(1x,d15.7))`
pub fn write_final_line(
    writer: &mut impl Write,
    sim_num: i32,
    values: &[f64; 52],
) -> io::Result<()> {
    write!(writer, " {:5}", sim_num)?;
    for val in values {
        write!(writer, " {}", fortran_float(*val, 15, 7))?;
    }
    writeln!(writer)?;
    Ok(())
}

/// Write fort.201 line (28 columns, D20.10 format).
///
/// Matches Fortran result.f: write(201,1001) format(28(1x,d20.10))
#[allow(dead_code)]
pub fn write_fort201_line(writer: &mut impl Write, values: &[f64; 28]) -> io::Result<()> {
    for val in values {
        write!(writer, " {}", fortran_float(*val, 20, 10))?;
    }
    writeln!(writer)?;
    Ok(())
}

/// Write fort.202 line (24 columns, D20.10 format).
#[allow(dead_code)]
pub fn write_fort202_line(writer: &mut impl Write, values: &[f64; 24]) -> io::Result<()> {
    for val in values {
        write!(writer, " {}", fortran_float(*val, 20, 10))?;
    }
    writeln!(writer)?;
    Ok(())
}

/// Create fort.201-204 output files and return buffered writers.
#[allow(dead_code)]
#[allow(clippy::type_complexity)]
pub fn create_fort_files(
    results_suffix: &str,
    output_dir: &str,
) -> io::Result<(
    BufWriter<File>,
    BufWriter<File>,
    BufWriter<File>,
    BufWriter<File>,
)> {
    let dir = format!("{}/resultats{}", output_dir, results_suffix);
    std::fs::create_dir_all(&dir).ok();

    let f201 = BufWriter::new(File::create("fort.201")?);
    let f202 = BufWriter::new(File::create("fort.202")?);
    let f203 = BufWriter::new(File::create("fort.203")?);
    let f204 = BufWriter::new(File::create("fort.204")?);

    Ok((f201, f202, f203, f204))
}
