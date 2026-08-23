use pyo3::prelude::*;

#[pyfunction]
fn version() -> PyResult<String> {
    Ok("Velo Native Core 0.1.0".to_string())
}

#[pymodule]
fn velo_native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
