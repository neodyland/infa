use std::io::{Read, Seek, SeekFrom};

#[derive(Debug, Clone)]
pub struct GGUF<R>
where
    R: Read,
{
    pub header: GGUFHeader,
    pub tensors: Vec<GGUFTensorMeta>,
    pub(super) tensor_bytes: R,
}

#[derive(Debug, Clone)]
pub struct GGUFTensor<'a> {
    pub data_type: &'a GGMLType,
    pub shape: &'a Vec<u64>,
    pub bytes: Vec<u8>,
}
impl<R> GGUF<R>
where
    R: Read,
{
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.iter().map(|t| t.name.clone()).collect()
    }
    pub fn get_tensor_meta(&mut self, name: &str) -> Option<&GGUFTensorMeta> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

impl<R> GGUF<R>
where
    R: Read + Seek,
{
    pub fn get_tensor<'a>(&'a mut self, name: &str) -> crate::Result<GGUFTensor<'a>> {
        if let Some(tensor) = self.tensors.iter().find(|t| t.name == name) {
            let start = tensor.offset;
            let size = tensor.data_type.size() * tensor.shape.iter().product::<u64>();
            let size = if size % 8 == 0 {
                size / 8
            } else {
                size / 8 + 1
            };
            let mut bytes = vec![0; size as usize];
            self.tensor_bytes.seek(SeekFrom::Start(start))?;
            self.tensor_bytes.read_exact(&mut bytes)?;
            Ok(GGUFTensor {
                data_type: &tensor.data_type,
                shape: &tensor.shape,
                bytes,
            })
        } else {
            Err(crate::Error::NoSuchTensor(name.to_string()))
        }
    }
}

#[derive(Debug, Clone)]
pub struct GGUFTensorMeta {
    pub name: String,
    pub data_type: GGMLType,
    pub shape: Vec<u64>,
    pub offset: u64,
}

#[derive(Debug, Clone)]
pub struct GGUFHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata_kv: Vec<GGUFMetadataKv>,
}
#[derive(Debug, Clone)]
pub struct GGUFMetadataKv {
    pub key: String,
    pub value_type: GGUFMetadataValueType,
    pub value: GGUFMetadataValue,
}
#[derive(Debug, Clone)]
pub enum GGUFMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFMetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone)]
pub enum GGUFMetadataValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for GGUFMetadataValueType {
    type Error = crate::Error;
    fn try_from(value: u32) -> crate::Result<Self> {
        Ok(match value {
            0 => GGUFMetadataValueType::Uint8,
            1 => GGUFMetadataValueType::Int8,
            2 => GGUFMetadataValueType::Uint16,
            3 => GGUFMetadataValueType::Int16,
            4 => GGUFMetadataValueType::Uint32,
            5 => GGUFMetadataValueType::Int32,
            6 => GGUFMetadataValueType::Float32,
            7 => GGUFMetadataValueType::Bool,
            8 => GGUFMetadataValueType::String,
            9 => GGUFMetadataValueType::Array,
            10 => GGUFMetadataValueType::Uint64,
            11 => GGUFMetadataValueType::Int64,
            12 => GGUFMetadataValueType::Float64,
            _ => return Err(crate::Error::InvalidGGUFMetadataValueType(value)),
        })
    }
}

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q4_2 = 4,
    Q4_3 = 5,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    Q4_0_4_4 = 31,
    Q4_0_4_8 = 32,
    Q4_0_8_8 = 33,
    TQ1_0 = 34,
    TQ2_0 = 35,
}

impl GGMLType {
    pub fn size(&self) -> u64 {
        match self {
            GGMLType::F64 | GGMLType::I64 => 64,
            GGMLType::F32 | GGMLType::I32 => 32,
            GGMLType::F16 | GGMLType::I16 | GGMLType::BF16 => 16,
            GGMLType::Q8_0 | GGMLType::Q8_1 | GGMLType::Q8_K | GGMLType::I8 => 8,
            GGMLType::Q6_K => 6,
            GGMLType::Q5_0 | GGMLType::Q5_1 | GGMLType::Q5_K => 5,
            GGMLType::Q4_0
            | GGMLType::Q4_1
            | GGMLType::Q4_2
            | GGMLType::Q4_3
            | GGMLType::Q4_K
            | GGMLType::IQ4_NL
            | GGMLType::IQ4_XS
            | GGMLType::Q4_0_4_4
            | GGMLType::Q4_0_4_8
            | GGMLType::Q4_0_8_8 => 4,
            GGMLType::Q3_K | GGMLType::IQ3_S | GGMLType::IQ3_XXS => 3,
            GGMLType::Q2_K
            | GGMLType::IQ2_XXS
            | GGMLType::IQ2_XS
            | GGMLType::IQ2_S
            | GGMLType::TQ2_0 => 2,
            GGMLType::IQ1_S | GGMLType::IQ1_M | GGMLType::TQ1_0 => 1,
        }
    }
}

impl TryFrom<u32> for GGMLType {
    type Error = crate::Error;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(GGMLType::F32),
            1 => Ok(GGMLType::F16),
            2 => Ok(GGMLType::Q4_0),
            3 => Ok(GGMLType::Q4_1),
            4 => Ok(GGMLType::Q4_2),
            5 => Ok(GGMLType::Q4_3),
            6 => Ok(GGMLType::Q5_0),
            7 => Ok(GGMLType::Q5_1),
            8 => Ok(GGMLType::Q8_0),
            9 => Ok(GGMLType::Q8_1),
            10 => Ok(GGMLType::Q2_K),
            11 => Ok(GGMLType::Q3_K),
            12 => Ok(GGMLType::Q4_K),
            13 => Ok(GGMLType::Q5_K),
            14 => Ok(GGMLType::Q6_K),
            15 => Ok(GGMLType::Q8_K),
            16 => Ok(GGMLType::IQ2_XXS),
            17 => Ok(GGMLType::IQ2_XS),
            18 => Ok(GGMLType::IQ3_XXS),
            19 => Ok(GGMLType::IQ1_S),
            20 => Ok(GGMLType::IQ4_NL),
            21 => Ok(GGMLType::IQ3_S),
            22 => Ok(GGMLType::IQ2_S),
            23 => Ok(GGMLType::IQ4_XS),
            24 => Ok(GGMLType::I8),
            25 => Ok(GGMLType::I16),
            26 => Ok(GGMLType::I32),
            27 => Ok(GGMLType::I64),
            28 => Ok(GGMLType::F64),
            29 => Ok(GGMLType::IQ1_M),
            30 => Ok(GGMLType::BF16),
            31 => Ok(GGMLType::Q4_0_4_4),
            32 => Ok(GGMLType::Q4_0_4_8),
            33 => Ok(GGMLType::Q4_0_8_8),
            34 => Ok(GGMLType::TQ1_0),
            35 => Ok(GGMLType::TQ2_0),
            _ => Err(crate::Error::InvalidGGMLType(value)),
        }
    }
}
