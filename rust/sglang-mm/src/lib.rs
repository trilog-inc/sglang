mod common;
mod dsv41;
mod inkling;
pub mod registry;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    common::register(m)?;
    inkling::register(m)?;
    dsv41::register(m)?;
    Ok(())
}
