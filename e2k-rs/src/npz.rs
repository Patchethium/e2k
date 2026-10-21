//! Minimal reader for the `.npz` model files written by `e2k_train/export.py`.
//!
//! Tensors are stored as little-endian `f16`/`f32`/`f64` `.npy` entries and
//! are always converted to `f32`. The `metadata` entry is a 0-d object array,
//! i.e. a pickled `ndarray` wrapping a `dict[str, str]`; a tiny pickle
//! interpreter below extracts that dict without needing Python.

use std::collections::HashMap;
use std::io::{Cursor, Read, Seek};

use ndarray::{Array1, Array2, ArrayD, IxDyn};

use crate::error::format_err;
use crate::Result;

pub(crate) struct Npz {
    tensors: HashMap<String, ArrayD<f32>>,
    pub metadata: HashMap<String, String>,
}

impl Npz {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_reader(Cursor::new(bytes))
    }

    pub fn from_reader<R: Read + Seek>(reader: R) -> Result<Self> {
        let mut archive = zip::ZipArchive::new(reader)?;
        let mut tensors = HashMap::new();
        let mut metadata = None;
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let name = file.name()?.trim_end_matches(".npy").to_string();
            let mut buf = Vec::with_capacity(file.size() as usize);
            file.read_to_end(&mut buf)?;
            let npy = parse_npy(&buf).map_err(|e| format_err(format!("{name}: {e}")))?;
            if name == "metadata" {
                metadata = Some(read_metadata(&npy)?);
            } else {
                tensors.insert(name, npy.to_f32()?);
            }
        }
        Ok(Self {
            tensors,
            metadata: metadata.ok_or_else(|| format_err("missing `metadata` entry"))?,
        })
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

struct Npy<'a> {
    descr: String,
    shape: Vec<usize>,
    data: &'a [u8],
}

fn parse_npy(buf: &[u8]) -> std::result::Result<Npy<'_>, String> {
    if buf.len() < 10 || &buf[..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let (header_len, offset) = match buf[6] {
        1 => (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10),
        2 | 3 => {
            let b = buf.get(8..12).ok_or("truncated header")?;
            (u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize, 12)
        }
        v => return Err(format!("unsupported .npy version {v}")),
    };
    let header = buf
        .get(offset..offset + header_len)
        .ok_or("truncated header")?;
    let header = std::str::from_utf8(header).map_err(|e| e.to_string())?;

    let descr = header_value(header, "descr")?
        .trim_matches(|c| c == '\'' || c == '"')
        .to_string();
    if header_value(header, "fortran_order")? != "False" {
        return Err("fortran order arrays are not supported".into());
    }
    let shape = header_value(header, "shape")?
        .trim_matches(|c| c == '(' || c == ')')
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|e| e.to_string()))
        .collect::<std::result::Result<_, _>>()?;
    Ok(Npy {
        descr,
        shape,
        data: &buf[offset + header_len..],
    })
}

/// Extracts the raw value of `key` from the python dict literal in a `.npy` header.
fn header_value<'a>(header: &'a str, key: &str) -> std::result::Result<&'a str, String> {
    let pat = format!("'{key}':");
    let start = header.find(&pat).ok_or(format!("header has no `{key}`"))? + pat.len();
    let rest = header[start..].trim_start();
    let end = if rest.starts_with('(') {
        rest.find(')').map(|i| i + 1)
    } else {
        rest.find(',')
    }
    .ok_or(format!("malformed `{key}` in header"))?;
    Ok(rest[..end].trim())
}

impl Npy<'_> {
    fn to_f32(&self) -> Result<ArrayD<f32>> {
        let n: usize = self.shape.iter().product();
        let values: Vec<f32> = match self.descr.as_str() {
            "<f2" => chunks::<2>(self.data, n)?
                .map(|b| half::f16::from_le_bytes(b).to_f32())
                .collect(),
            "<f4" => chunks::<4>(self.data, n)?.map(f32::from_le_bytes).collect(),
            "<f8" => chunks::<8>(self.data, n)?
                .map(|b| f64::from_le_bytes(b) as f32)
                .collect(),
            d => return Err(format_err(format!("unsupported dtype `{d}`"))),
        };
        ArrayD::from_shape_vec(IxDyn(&self.shape), values).map_err(|e| format_err(e.to_string()))
    }
}

fn chunks<const N: usize>(data: &[u8], n: usize) -> Result<impl Iterator<Item = [u8; N]> + '_> {
    if data.len() < n * N {
        return Err(format_err("truncated array data"));
    }
    Ok(data[..n * N].as_chunks::<N>().0.iter().copied())
}

fn read_metadata(npy: &Npy) -> Result<HashMap<String, String>> {
    if npy.descr != "|O" {
        return Err(format_err("`metadata` is not an object array"));
    }
    let value = pickle::load(npy.data).map_err(|e| format_err(format!("metadata: {e}")))?;
    let dict = value
        .find_dict()
        .ok_or_else(|| format_err("metadata does not contain a dict"))?;
    Ok(dict
        .iter()
        .filter_map(|(k, v)| match (k, v) {
            (pickle::Value::Str(k), pickle::Value::Str(v)) => Some((k.clone(), v.clone())),
            _ => None,
        })
        .collect())
}

/// Just enough of the pickle protocol (2-5) to read a pickled numpy object array.
mod pickle {
    use std::collections::HashMap;

    // Scalars are kept for fidelity even though only strings are read back.
    #[allow(dead_code)]
    #[derive(Clone, Debug)]
    pub enum Value {
        None,
        Bool(bool),
        Int(i64),
        Float(f64),
        Str(String),
        Bytes(Vec<u8>),
        List(Vec<Value>),
        Tuple(Vec<Value>),
        Dict(Vec<(Value, Value)>),
        Global,
        /// The result of `REDUCE`/`NEWOBJ`, with whatever `BUILD` attached to it.
        Object {
            args: Box<Value>,
            state: Box<Value>,
        },
    }

    impl Value {
        /// Depth-first search for the first dict in the object graph.
        pub fn find_dict(&self) -> Option<&Vec<(Value, Value)>> {
            match self {
                Value::Dict(d) => Some(d),
                Value::List(v) | Value::Tuple(v) => v.iter().find_map(Value::find_dict),
                Value::Object { args, state } => state.find_dict().or_else(|| args.find_dict()),
                _ => None,
            }
        }
    }

    enum Item {
        Mark,
        Value(Value),
    }

    struct Reader<'a> {
        data: &'a [u8],
        pos: usize,
    }

    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
            let end = self.pos.checked_add(n).filter(|&e| e <= self.data.len());
            let end = end.ok_or("unexpected end of pickle")?;
            let out = &self.data[self.pos..end];
            self.pos = end;
            Ok(out)
        }

        fn u8(&mut self) -> Result<u8, String> {
            Ok(self.take(1)?[0])
        }

        fn bytes<const N: usize>(&mut self) -> Result<[u8; N], String> {
            Ok(self.take(N)?.try_into().expect("slice has N bytes"))
        }

        fn line(&mut self) -> Result<String, String> {
            let rest = &self.data[self.pos..];
            let end = rest
                .iter()
                .position(|&b| b == b'\n')
                .ok_or("unterminated line")?;
            let s = String::from_utf8_lossy(&rest[..end]).into_owned();
            self.pos += end + 1;
            Ok(s)
        }

        fn string(&mut self, len: usize) -> Result<Value, String> {
            let bytes = self.take(len)?;
            Ok(Value::Str(
                String::from_utf8(bytes.to_vec()).map_err(|e| e.to_string())?,
            ))
        }
    }

    pub fn load(data: &[u8]) -> Result<Value, String> {
        let mut r = Reader { data, pos: 0 };
        let mut stack: Vec<Item> = Vec::new();
        // Values are copied into the memo, which is fine for the acyclic,
        // build-once object graphs numpy produces.
        let mut memo: HashMap<u32, Value> = HashMap::new();

        fn pop(stack: &mut Vec<Item>) -> Result<Value, String> {
            match stack.pop() {
                Some(Item::Value(v)) => Ok(v),
                _ => Err("stack underflow".into()),
            }
        }
        fn pop_mark(stack: &mut Vec<Item>) -> Result<Vec<Value>, String> {
            let mark = stack
                .iter()
                .rposition(|i| matches!(i, Item::Mark))
                .ok_or("missing mark")?;
            let items = stack.split_off(mark + 1);
            stack.pop();
            Ok(items
                .into_iter()
                .map(|i| match i {
                    Item::Value(v) => v,
                    Item::Mark => unreachable!(),
                })
                .collect())
        }
        fn top(stack: &mut [Item]) -> Result<&mut Value, String> {
            match stack.last_mut() {
                Some(Item::Value(v)) => Ok(v),
                _ => Err("stack underflow".into()),
            }
        }
        fn pairs(items: Vec<Value>) -> Result<Vec<(Value, Value)>, String> {
            if !items.len().is_multiple_of(2) {
                return Err("odd number of dict items".into());
            }
            let mut it = items.into_iter();
            let mut out = Vec::new();
            while let (Some(k), Some(v)) = (it.next(), it.next()) {
                out.push((k, v));
            }
            Ok(out)
        }

        loop {
            let op = r.u8()?;
            let value = match op {
                0x80 => {
                    r.u8()?; // PROTO
                    continue;
                }
                0x95 => {
                    r.bytes::<8>()?; // FRAME
                    continue;
                }
                b'.' => return pop(&mut stack),
                b'(' => {
                    stack.push(Item::Mark);
                    continue;
                }
                0x94 => {
                    let v = match stack.last() {
                        Some(Item::Value(v)) => v.clone(),
                        _ => return Err("MEMOIZE on empty stack".into()),
                    };
                    memo.insert(memo.len() as u32, v);
                    continue;
                }
                b'q' | b'r' => {
                    let idx = if op == b'q' {
                        r.u8()? as u32
                    } else {
                        u32::from_le_bytes(r.bytes()?)
                    };
                    let v = match stack.last() {
                        Some(Item::Value(v)) => v.clone(),
                        _ => return Err("PUT on empty stack".into()),
                    };
                    memo.insert(idx, v);
                    continue;
                }
                b'h' | b'j' => {
                    let idx = if op == b'h' {
                        r.u8()? as u32
                    } else {
                        u32::from_le_bytes(r.bytes()?)
                    };
                    memo.get(&idx).cloned().ok_or("bad memo index")?
                }
                b'N' => Value::None,
                0x88 => Value::Bool(true),
                0x89 => Value::Bool(false),
                b'K' => Value::Int(r.u8()? as i64),
                b'M' => Value::Int(u16::from_le_bytes(r.bytes()?) as i64),
                b'J' => Value::Int(i32::from_le_bytes(r.bytes()?) as i64),
                b'G' => Value::Float(f64::from_be_bytes(r.bytes()?)),
                0x8c => {
                    let len = r.u8()? as usize;
                    r.string(len)?
                }
                b'X' => {
                    let len = u32::from_le_bytes(r.bytes()?) as usize;
                    r.string(len)?
                }
                0x8d => {
                    let len = u64::from_le_bytes(r.bytes()?) as usize;
                    r.string(len)?
                }
                b'C' => {
                    let len = r.u8()? as usize;
                    Value::Bytes(r.take(len)?.to_vec())
                }
                b'B' => {
                    let len = u32::from_le_bytes(r.bytes()?) as usize;
                    Value::Bytes(r.take(len)?.to_vec())
                }
                b'c' => {
                    r.line()?;
                    r.line()?;
                    Value::Global
                }
                0x93 => {
                    pop(&mut stack)?;
                    pop(&mut stack)?;
                    Value::Global
                }
                b'R' | 0x81 => {
                    let args = pop(&mut stack)?;
                    pop(&mut stack)?;
                    Value::Object {
                        args: Box::new(args),
                        state: Box::new(Value::None),
                    }
                }
                b'b' => {
                    let new_state = pop(&mut stack)?;
                    if let Value::Object { state, .. } = top(&mut stack)? {
                        **state = new_state;
                    }
                    continue;
                }
                b')' => Value::Tuple(vec![]),
                b']' => Value::List(vec![]),
                b'}' => Value::Dict(vec![]),
                b't' => Value::Tuple(pop_mark(&mut stack)?),
                b'l' => Value::List(pop_mark(&mut stack)?),
                b'd' => Value::Dict(pairs(pop_mark(&mut stack)?)?),
                0x85..=0x87 => {
                    let n = (op - 0x84) as usize;
                    let mut items = (0..n)
                        .map(|_| pop(&mut stack))
                        .collect::<Result<Vec<_>, _>>()?;
                    items.reverse();
                    Value::Tuple(items)
                }
                b'a' => {
                    let v = pop(&mut stack)?;
                    match top(&mut stack)? {
                        Value::List(l) => l.push(v),
                        _ => return Err("APPEND to non-list".into()),
                    }
                    continue;
                }
                b'e' => {
                    let items = pop_mark(&mut stack)?;
                    match top(&mut stack)? {
                        Value::List(l) => l.extend(items),
                        _ => return Err("APPENDS to non-list".into()),
                    }
                    continue;
                }
                b's' | b'u' => {
                    let items = if op == b's' {
                        let v = pop(&mut stack)?;
                        let k = pop(&mut stack)?;
                        vec![k, v]
                    } else {
                        pop_mark(&mut stack)?
                    };
                    match top(&mut stack)? {
                        Value::Dict(d) => d.extend(pairs(items)?),
                        _ => return Err("SETITEM on non-dict".into()),
                    }
                    continue;
                }
                op => return Err(format!("unsupported pickle opcode 0x{op:02x}")),
            };
            stack.push(Item::Value(value));
        }
    }
}
