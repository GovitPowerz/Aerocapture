//! Named-tensor storage behind the per-layer tensor table.
//!
//! A layer declares its weights once with `tensor_table!`; the listed fields,
//! in that order, ARE the canonical flat order (PSO chromosome) and the JSON
//! weight keys. Everything that reads or writes weights -- `to_flat`,
//! `from_flat`, `n_params`, the JSON codec, the PyO3 schema accessor -- walks
//! the table through the `Tensor` trait below, so a new layer type adds a
//! struct, a forward pass, and one `tensor_table!` line.

use nalgebra::{DMatrix, DVector};
use serde_json::Value;

/// Shape of one tensor. Matrices are row-major in the flat order and a list
/// of rows in JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Scalar,
    Vec(usize),
    Mat(usize, usize),
}

impl Shape {
    pub fn numel(self) -> usize {
        match self {
            Shape::Scalar => 1,
            Shape::Vec(n) => n,
            Shape::Mat(r, c) => r * c,
        }
    }

    /// `[]` scalar, `[n]` vector, `[rows, cols]` matrix (the PyO3 schema form).
    pub fn dims(self) -> Vec<usize> {
        match self {
            Shape::Scalar => vec![],
            Shape::Vec(n) => vec![n],
            Shape::Mat(r, c) => vec![r, c],
        }
    }
}

/// One weight tensor of a layer: row-major flat read/write plus its JSON view.
pub trait Tensor {
    fn shape(&self) -> Shape;
    fn write_flat(&self, out: &mut Vec<f64>);
    /// `src.len() == self.shape().numel()`; callers slice exactly.
    fn read_flat(&mut self, src: &[f64]);
    fn to_json(&self) -> Value;
}

fn num(x: f64) -> Value {
    // Non-finite -> null, matching serde's own f64 serialization.
    serde_json::Number::from_f64(x).map_or(Value::Null, Value::Number)
}

fn json_row(v: &[f64]) -> Value {
    Value::Array(v.iter().map(|&x| num(x)).collect())
}

impl Tensor for f64 {
    fn shape(&self) -> Shape {
        Shape::Scalar
    }
    fn write_flat(&self, out: &mut Vec<f64>) {
        out.push(*self);
    }
    fn read_flat(&mut self, src: &[f64]) {
        *self = src[0];
    }
    fn to_json(&self) -> Value {
        num(*self)
    }
}

impl Tensor for Vec<f64> {
    fn shape(&self) -> Shape {
        Shape::Vec(self.len())
    }
    fn write_flat(&self, out: &mut Vec<f64>) {
        out.extend_from_slice(self);
    }
    fn read_flat(&mut self, src: &[f64]) {
        self.copy_from_slice(src);
    }
    fn to_json(&self) -> Value {
        json_row(self)
    }
}

impl Tensor for Vec<Vec<f64>> {
    fn shape(&self) -> Shape {
        Shape::Mat(self.len(), self.first().map_or(0, Vec::len))
    }
    fn write_flat(&self, out: &mut Vec<f64>) {
        for row in self {
            out.extend_from_slice(row);
        }
    }
    fn read_flat(&mut self, src: &[f64]) {
        let mut idx = 0;
        for row in self.iter_mut() {
            let n = row.len();
            row.copy_from_slice(&src[idx..idx + n]);
            idx += n;
        }
    }
    fn to_json(&self) -> Value {
        Value::Array(self.iter().map(|r| json_row(r)).collect())
    }
}

impl Tensor for DVector<f64> {
    fn shape(&self) -> Shape {
        Shape::Vec(self.len())
    }
    fn write_flat(&self, out: &mut Vec<f64>) {
        out.extend(self.iter().copied());
    }
    fn read_flat(&mut self, src: &[f64]) {
        *self = DVector::from_row_slice(src);
    }
    fn to_json(&self) -> Value {
        json_row(self.as_slice())
    }
}

impl Tensor for DMatrix<f64> {
    fn shape(&self) -> Shape {
        Shape::Mat(self.nrows(), self.ncols())
    }
    fn write_flat(&self, out: &mut Vec<f64>) {
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                out.push(self[(i, j)]);
            }
        }
    }
    fn read_flat(&mut self, src: &[f64]) {
        *self = DMatrix::from_row_slice(self.nrows(), self.ncols(), src);
    }
    fn to_json(&self) -> Value {
        Value::Array(
            (0..self.nrows())
                .map(|i| Value::Array((0..self.ncols()).map(|j| num(self[(i, j)])).collect()))
                .collect(),
        )
    }
}

/// A struct field that may contribute a tensor to the table. `Option` fields
/// (flag-gated tensors such as Mamba3's `a_imag`) contribute only when `Some`,
/// so presence is fixed at construction and never re-decided by a load.
pub trait TensorField {
    fn as_tensor(&self) -> Option<&dyn Tensor>;
    fn as_tensor_mut(&mut self) -> Option<&mut dyn Tensor>;
}

macro_rules! tensor_field_for {
    ($($t:ty),* $(,)?) => {$(
        impl TensorField for $t {
            fn as_tensor(&self) -> Option<&dyn Tensor> {
                Some(self)
            }
            fn as_tensor_mut(&mut self) -> Option<&mut dyn Tensor> {
                Some(self)
            }
        }
    )*};
}
tensor_field_for!(f64, Vec<f64>, Vec<Vec<f64>>, DVector<f64>, DMatrix<f64>);

impl<T: Tensor> TensorField for Option<T> {
    fn as_tensor(&self) -> Option<&dyn Tensor> {
        self.as_ref().map(|t| t as &dyn Tensor)
    }
    fn as_tensor_mut(&mut self) -> Option<&mut dyn Tensor> {
        self.as_mut().map(|t| t as &mut dyn Tensor)
    }
}

/// Declare a layer's tensor table and derive its `LayerWeights` impl.
///
/// The listed fields, in this order, are the canonical flat order and the
/// JSON weight keys (`to_flat` / `from_flat` / `n_params` come from the trait
/// defaults). `post_load = method` names the derived-field hook run at the end
/// of every load (e.g. the Transformer PE offsets).
macro_rules! tensor_table {
    ($ty:ty { $($field:ident),* $(,)? } $(, post_load = $hook:ident)? $(,)?) => {
        impl $crate::data::neural::LayerWeights for $ty {
            fn tensors(&self) -> Vec<(&'static str, &dyn $crate::data::neural::Tensor)> {
                std::iter::empty::<(&'static str, &dyn $crate::data::neural::Tensor)>()
                    $(.chain(
                        $crate::data::neural::TensorField::as_tensor(&self.$field)
                            .map(|t| (stringify!($field), t)),
                    ))*
                    .collect()
            }

            fn tensors_mut(&mut self) -> Vec<(&'static str, &mut dyn $crate::data::neural::Tensor)> {
                std::iter::empty::<(&'static str, &mut dyn $crate::data::neural::Tensor)>()
                    $(.chain(
                        $crate::data::neural::TensorField::as_tensor_mut(&mut self.$field)
                            .map(|t| (stringify!($field), t)),
                    ))*
                    .collect()
            }

            $(
                fn post_load(&mut self) {
                    self.$hook();
                }
            )?
        }
    };
}
pub(crate) use tensor_table;

/// Append a JSON tensor of the given shape to `out`, row-major, shape-checked.
pub fn json_to_flat(v: &Value, shape: Shape, out: &mut Vec<f64>) -> Result<(), String> {
    fn number(v: &Value) -> Result<f64, String> {
        v.as_f64().ok_or_else(|| "expected a number".to_string())
    }
    fn row(v: &Value, n: usize, out: &mut Vec<f64>) -> Result<(), String> {
        let a = v
            .as_array()
            .ok_or_else(|| format!("expected a list of {n} numbers"))?;
        if a.len() != n {
            return Err(format!("expected length {n}, got {}", a.len()));
        }
        for x in a {
            out.push(number(x)?);
        }
        Ok(())
    }
    match shape {
        Shape::Scalar => out.push(number(v)?),
        Shape::Vec(n) => row(v, n, out)?,
        Shape::Mat(r, c) => {
            let rows = v
                .as_array()
                .ok_or_else(|| format!("expected {r} rows of {c} numbers"))?;
            if rows.len() != r {
                return Err(format!("expected {r} rows, got {}", rows.len()));
            }
            for (k, rv) in rows.iter().enumerate() {
                row(rv, c, out).map_err(|e| format!("row {k}: {e}"))?;
            }
        }
    }
    Ok(())
}
