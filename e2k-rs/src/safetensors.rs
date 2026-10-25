//! Minimal reader for the `.safetensors` model files written by `e2k_train/export.py`.
//!
//! The file is a little-endian `u64` header length, a JSON header mapping each
//! tensor to its `dtype`, `shape` and `data_offsets` (plus a `__metadata__`
//! map of strings), then the raw tensor data. Tensors are stored as
//! little-endian `f16`/`f32`/`f64` and are always converted to `f32`.

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use serde_json::{Map, Value};

use crate::error::format_err;
use crate::Result;

pub(crate) struct SafeTensors {
    tensors: HashMap<String, ArrayD<f32>>,
    metadata: HashMap<String, String>,
}

impl SafeTensors {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let len = bytes
            .first_chunk::<8>()
            .map(|b| u64::from_le_bytes(*b) as usize)
            .ok_or_else(|| format_err("truncated header"))?;
        let header = bytes
            .get(8..8usize.saturating_add(len))
            .ok_or_else(|| format_err("truncated header"))?;
        let mut header: Map<String, Value> = serde_json::from_slice(header)?;
        let body = &bytes[8 + len..];

        let metadata = match header.remove("__metadata__") {
            Some(Value::Object(meta)) => meta
                .into_iter()
                .map(|(k, v)| match v {
                    Value::String(v) => Ok((k, v)),
                    _ => Err(format_err(format!("metadata `{k}` is not a string"))),
                })
                .collect::<Result<_>>()?,
            Some(_) => return Err(format_err("`__metadata__` is not an object")),
            None => HashMap::new(),
        };
        let tensors = header
            .iter()
            .map(|(name, info)| {
                let tensor =
                    read_tensor(info, body).map_err(|e| format_err(format!("{name}: {e}")))?;
                Ok((name.clone(), tensor))
            })
            .collect::<Result<_>>()?;
        Ok(Self { tensors, metadata })
    }

    fn get(&self, name: &str) -> Result<&ArrayD<f32>> {
        self.tensors
            .get(name)
            .ok_or_else(|| format_err(format!("missing tensor `{name}`")))
    }

    pub fn array1(&self, name: &str) -> Result<Array1<f32>> {
        self.get(name)?
            .clone()
            .into_dimensionality()
            .map_err(|e| format_err(format!("tensor `{name}` is not 1-d: {e}")))
    }

    pub fn array2(&self, name: &str) -> Result<Array2<f32>> {
        self.get(name)?
            .clone()
            .into_dimensionality()
            .map_err(|e| format_err(format!("tensor `{name}` is not 2-d: {e}")))
    }

    pub fn meta(&self, key: &str) -> Result<&str> {
        self.metadata
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| format_err(format!("missing metadata `{key}`")))
    }
}

fn read_tensor(info: &Value, body: &[u8]) -> std::result::Result<ArrayD<f32>, String> {
    let dtype = info["dtype"].as_str().ok_or("`dtype` is not a string")?;
    let shape = info["shape"]
        .as_array()
        .ok_or("`shape` is not an array")?
        .iter()
        .map(|d| d.as_u64().map(|d| d as usize).ok_or("bad `shape`"))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let offsets = info["data_offsets"]
        .as_array()
        .and_then(|o| Some((o.first()?.as_u64()?, o.get(1)?.as_u64()?)))
        .ok_or("bad `data_offsets`")?;
    let data = body
        .get(offsets.0 as usize..offsets.1 as usize)
        .ok_or("`data_offsets` out of range")?;

    let n: usize = shape.iter().product();
    let values: Vec<f32> = match dtype {
        "F16" => chunks::<2>(data, n)?
            .map(|b| half::f16::from_le_bytes(b).to_f32())
            .collect(),
        "F32" => chunks::<4>(data, n)?.map(f32::from_le_bytes).collect(),
        "F64" => chunks::<8>(data, n)?
            .map(|b| f64::from_le_bytes(b) as f32)
            .collect(),
        d => return Err(format!("unsupported dtype `{d}`")),
    };
    ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|e| e.to_string())
}

fn chunks<const N: usize>(
    data: &[u8],
    n: usize,
) -> std::result::Result<impl Iterator<Item = [u8; N]> + '_, String> {
    if data.len() != n * N {
        return Err("data size does not match `shape`".into());
    }
    Ok(data.as_chunks::<N>().0.iter().copied())
}
