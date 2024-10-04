mod typing;

use std::io::Read;

pub use typing::*;

pub struct GGUFParser<R>
where
    R: Read,
{
    bytes: R,
    offset: usize,
}

impl<R> GGUFParser<R>
where
    R: Read,
{
    pub fn new(bytes: R) -> Self {
        Self { bytes, offset: 0 }
    }
    fn read_bytes(&mut self, len: usize) -> crate::Result<Vec<u8>> {
        let mut buf = vec![0; len];
        self.bytes.read_exact(&mut buf)?;
        self.offset += len;
        Ok(buf)
    }
    fn read_string(&mut self) -> crate::Result<String> {
        let len = self.read_bytes(8)?;
        let len = u64::from_le_bytes(len.try_into().unwrap()) as usize;
        let str_bytes = self.read_bytes(len)?;
        Ok(String::from_utf8(str_bytes)?)
    }
    fn read_metadata_value(
        &mut self,
        ty: &GGUFMetadataValueType,
    ) -> crate::Result<GGUFMetadataValue> {
        Ok(match ty {
            GGUFMetadataValueType::Bool => GGUFMetadataValue::Bool(self.read_bytes(1)?[0] != 0),
            GGUFMetadataValueType::Uint8 => GGUFMetadataValue::Uint8(self.read_bytes(1)?[0]),
            GGUFMetadataValueType::Int8 => GGUFMetadataValue::Int8(i8::from_le_bytes(
                self.read_bytes(1)?[0..1].try_into().unwrap(),
            )),
            GGUFMetadataValueType::Uint16 => GGUFMetadataValue::Uint16(u16::from_le_bytes(
                self.read_bytes(2)?[0..2].try_into().unwrap(),
            )),
            GGUFMetadataValueType::Int16 => GGUFMetadataValue::Int16(i16::from_le_bytes(
                self.read_bytes(2)?[0..2].try_into().unwrap(),
            )),
            GGUFMetadataValueType::Uint32 => GGUFMetadataValue::Uint32(u32::from_le_bytes(
                self.read_bytes(4)?[0..4].try_into().unwrap(),
            )),
            GGUFMetadataValueType::Int32 => GGUFMetadataValue::Int32(i32::from_le_bytes(
                self.read_bytes(4)?[0..4].try_into().unwrap(),
            )),
            GGUFMetadataValueType::String => GGUFMetadataValue::String(self.read_string()?),
            GGUFMetadataValueType::Array => {
                let ty = self.read_bytes(4)?;
                let ty =
                    GGUFMetadataValueType::try_from(u32::from_le_bytes(ty.try_into().unwrap()))?;
                let len = self.read_bytes(8)?;
                let len = u64::from_le_bytes(len.try_into().unwrap());
                let mut values = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    values.push(self.read_metadata_value(&ty)?);
                }
                GGUFMetadataValue::Array(values)
            }
            GGUFMetadataValueType::Uint64 => GGUFMetadataValue::Uint64(u64::from_le_bytes(
                self.read_bytes(8)?[0..8].try_into().unwrap(),
            )),
            GGUFMetadataValueType::Int64 => GGUFMetadataValue::Int64(i64::from_le_bytes(
                self.read_bytes(8)?[0..8].try_into().unwrap(),
            )),
            GGUFMetadataValueType::Float64 => {
                let bytes = self.read_bytes(8)?;
                let value = f64::from_le_bytes(bytes.try_into().unwrap());
                GGUFMetadataValue::Float64(value)
            }
            GGUFMetadataValueType::Float32 => {
                let bytes = self.read_bytes(4)?;
                let value = f32::from_le_bytes(bytes.try_into().unwrap());
                GGUFMetadataValue::Float32(value)
            }
        })
    }
    fn read_metadata_kv(&mut self) -> crate::Result<GGUFMetadataKv> {
        let key = self.read_string()?;
        let value_type = self.read_bytes(4)?;
        let value_type = u32::from_le_bytes(value_type.try_into().unwrap());
        let value_type = GGUFMetadataValueType::try_from(value_type)?;
        let value = self.read_metadata_value(&value_type)?;
        Ok(GGUFMetadataKv {
            key,
            value_type,
            value,
        })
    }
    fn read_header(&mut self) -> crate::Result<GGUFHeader> {
        let magic = self.read_bytes(4)?;
        if magic != b"GGUF" {
            return Err(crate::Error::MagicMismatch);
        }
        let version = self.read_bytes(4)?;
        let version = u32::from_le_bytes(version.try_into().unwrap());
        let tensor_count = self.read_bytes(8)?;
        let tensor_count = u64::from_le_bytes(tensor_count.try_into().unwrap());
        let metadata_kv_count = self.read_bytes(8)?;
        let metadata_kv_count = u64::from_le_bytes(metadata_kv_count.try_into().unwrap());
        let mut metadata_kv = Vec::with_capacity(metadata_kv_count as usize);
        for _ in 0..metadata_kv_count {
            metadata_kv.push(self.read_metadata_kv()?);
        }
        Ok(GGUFHeader {
            version,
            tensor_count,
            metadata_kv_count,
            metadata_kv,
        })
    }
    fn read_tensor(&mut self, alginment: u64) -> crate::Result<GGUFTensorMeta> {
        let name = self.read_string()?;
        let n_dimensions = self.read_bytes(4)?;
        let n_dimensions = u32::from_le_bytes(n_dimensions.try_into().unwrap());
        let mut shape = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            let dim = self.read_bytes(8)?;
            shape.push(u64::from_le_bytes(dim.try_into().unwrap()));
        }
        let data_type = self.read_bytes(4)?;
        let data_type = u32::from_le_bytes(data_type.try_into().unwrap());
        let offset = self.read_bytes(8)?;
        let offset = u64::from_le_bytes(offset.try_into().unwrap());
        let offset = offset + (alginment - (offset % alginment)) % alginment;
        Ok(GGUFTensorMeta {
            name,
            shape,
            data_type: GGMLType::try_from(data_type)?,
            offset,
        })
    }
    pub fn parse(mut self) -> crate::Result<GGUF<R>> {
        let header = self.read_header()?;
        let alignment = header
            .metadata_kv
            .iter()
            .find(|x| x.key == "general.alignment")
            .map(|x| x.value.clone())
            .unwrap_or(GGUFMetadataValue::Uint32(32));
        let alignment = match alignment {
            GGUFMetadataValue::Uint32(v) => v,
            _ => return Err(crate::Error::InvalidAlignmentMetaType(alignment)),
        } as u64;
        let mut tensors = Vec::with_capacity(header.tensor_count as usize);
        for _ in 0..header.tensor_count {
            tensors.push(self.read_tensor(alignment)?);
        }
        let alignment = alignment as usize;
        self.offset += (alignment - (self.offset % alignment)) % alignment;
        Ok(GGUF {
            header,
            tensors,
            tensor_bytes: self.bytes,
            offset: self.offset as u64,
        })
    }
}
