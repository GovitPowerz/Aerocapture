use pyo3::prelude::*;

/// Aerocapture trajectory simulator Python bindings.
#[pymodule]
fn aerocapture_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    Ok(())
}
