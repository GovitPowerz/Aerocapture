//! Neural network parameters loader.
//!
//! Matches Fortran lecgnn.f — reads nn_param{suffix} file.
//! Network architecture from param_algo: 6 inputs → 12 hidden (tanh) → 2 outputs (asinh).

use super::DataError;

/// NN architecture constants (matching Fortran param_algo).
pub const N_INPUT: usize = 6;
pub const N_HIDDEN: usize = 12;
pub const N_OUTPUT: usize = 2;

/// Neural network weight parameters.
#[derive(Debug, Clone)]
pub struct NeuralNetParams {
    /// Input→hidden weights [N_HIDDEN x N_INPUT] (row-major: lw1[j][i])
    pub lw1: [[f64; N_INPUT]; N_HIDDEN],
    /// Hidden biases [N_HIDDEN]
    pub bias1: [f64; N_HIDDEN],
    /// Hidden→output weights [N_OUTPUT x N_HIDDEN] (row-major: lw4[j][i])
    pub lw4: [[f64; N_HIDDEN]; N_OUTPUT],
    /// Output biases [N_OUTPUT]
    pub bias4: [f64; N_OUTPUT],
}

impl NeuralNetParams {
    /// Load NN weights from file.
    ///
    /// Matches Fortran lecgnn.f reading order:
    /// - Skip 6 header lines
    /// - Read n1lw1(j,i) for i=1..n_input, j=1..n_hidden (column-major)
    /// - Read n1bias1(i) for i=1..n_hidden
    /// - Read n1lw4(j,i) for i=1..n_hidden, j=1..n_output (column-major)
    /// - Read n1bias4(i) for i=1..n_output
    pub fn load(path: &str) -> Result<Self, DataError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| DataError(format!("Cannot read {}: {}", path, e)))?;

        let values: Vec<f64> = content
            .lines()
            .skip(6) // 6 header lines
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let token = l.trim().split_whitespace().next().unwrap_or("0");
                token.parse::<f64>().map_err(|_| {
                    DataError(format!("Cannot parse '{}' as f64 in {}", token, path))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let expected = N_INPUT * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_OUTPUT + N_OUTPUT;
        if values.len() < expected {
            return Err(DataError(format!(
                "NN param file too short: {} values, need {} in {}",
                values.len(),
                expected,
                path
            )));
        }

        let mut idx = 0;
        let mut lw1 = [[0.0; N_INPUT]; N_HIDDEN];
        // Fortran reads: do i=1,n_input; do j=1,n_hidden; read n1lw1(j,i)
        for i in 0..N_INPUT {
            for j in 0..N_HIDDEN {
                lw1[j][i] = values[idx];
                idx += 1;
            }
        }

        let mut bias1 = [0.0; N_HIDDEN];
        for j in 0..N_HIDDEN {
            bias1[j] = values[idx];
            idx += 1;
        }

        let mut lw4 = [[0.0; N_HIDDEN]; N_OUTPUT];
        // Fortran reads: do i=1,n_hidden; do j=1,n_output; read n1lw4(j,i)
        for i in 0..N_HIDDEN {
            for j in 0..N_OUTPUT {
                lw4[j][i] = values[idx];
                idx += 1;
            }
        }

        let mut bias4 = [0.0; N_OUTPUT];
        for j in 0..N_OUTPUT {
            bias4[j] = values[idx];
            idx += 1;
        }

        Ok(NeuralNetParams {
            lw1,
            bias1,
            lw4,
            bias4,
        })
    }
}
