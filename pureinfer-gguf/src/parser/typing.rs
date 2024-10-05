use std::io::{Read, Seek, SeekFrom};

use half::{bf16, f16};

#[derive(Debug, Clone)]
pub struct GGUF<R>
where
    R: Read,
{
    pub header: GGUFHeader,
    pub tensors: Vec<GGUFTensorMeta>,
    pub(super) tensor_bytes: R,
    pub(super) offset: u64,
}

pub struct GGUFTensor<'a> {
    pub shape: &'a Vec<u64>,
    pub bytes: Box<dyn crate::BaseGGUFBlock>,
    pub data_type: &'a GGMLType,
}

impl<'a> GGUFTensor<'a> {
    pub fn from_raw_parts<T>(
        bytes: Vec<u8>,
        size: usize,
        shape: &'a Vec<u64>,
        data_type: &'a GGMLType,
    ) -> Self
    where
        T: crate::GGUFBlock + 'static,
    {
        let raw_data_ptr = bytes.as_ptr();
        let n_blocks = size / std::mem::size_of::<T>();
        let bytes = unsafe { std::slice::from_raw_parts(raw_data_ptr as *const T, n_blocks) };
        Self {
            shape,
            bytes: Box::new(bytes),
            data_type,
        }
    }
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
            let size = tensor.data_type.size() * tensor.shape.iter().product::<u64>() as usize
                / tensor.data_type.block_size();
            let mut bytes = vec![0; size as usize];
            self.tensor_bytes
                .seek(SeekFrom::Start(start + self.offset))?;
            self.tensor_bytes.read_exact(&mut bytes)?;
            Ok(match tensor.data_type {
                GGMLType::Q4_0 => GGUFTensor::from_raw_parts::<crate::BlockQ4_0>(
                    bytes,
                    size,
                    &tensor.shape,
                    &tensor.data_type,
                ),
                GGMLType::Q4_1 => GGUFTensor::from_raw_parts::<crate::BlockQ4_1>(
                    bytes,
                    size,
                    &tensor.shape,
                    &tensor.data_type,
                ),
                GGMLType::Q5_0 => GGUFTensor::from_raw_parts::<crate::BlockQ5_0>(
                    bytes,
                    size,
                    &tensor.shape,
                    &tensor.data_type,
                ),
                GGMLType::Q5_1 => GGUFTensor::from_raw_parts::<crate::BlockQ5_1>(
                    bytes,
                    size,
                    &tensor.shape,
                    &tensor.data_type,
                ),
                GGMLType::Q8_0 => GGUFTensor::from_raw_parts::<crate::BlockQ8_0>(
                    bytes,
                    size,
                    &tensor.shape,
                    &tensor.data_type,
                ),
                GGMLType::Q8_1 => GGUFTensor::from_raw_parts::<crate::BlockQ8_1>(
                    bytes,
                    size,
                    &tensor.shape,
                    &tensor.data_type,
                ),
                GGMLType::Q2_K => GGUFTensor::from_raw_parts::<crate::BlockQ2K>(
                    bytes,
                    size,
                    &tensor.shape,
                    &tensor.data_type,
                ),
                GGMLType::Q3_K => GGUFTensor::from_raw_parts::<crate::BlockQ3K>(
                    bytes,
                    size,
                    &tensor.shape,
                    &tensor.data_type,
                ),
                GGMLType::Q4_K => GGUFTensor::from_raw_parts::<crate::BlockQ4K>(
                    bytes,
                    size,
                    &tensor.shape,
                    &tensor.data_type,
                ),
                GGMLType::Q5_K => GGUFTensor::from_raw_parts::<crate::BlockQ5K>(
                    bytes,
                    size,
                    &tensor.shape,
                    &tensor.data_type,
                ),
                GGMLType::Q6_K => GGUFTensor::from_raw_parts::<crate::BlockQ6K>(
                    bytes,
                    size,
                    &tensor.shape,
                    &tensor.data_type,
                ),
                GGMLType::Q8_K => GGUFTensor::from_raw_parts::<crate::BlockQ8K>(
                    bytes,
                    size,
                    &tensor.shape,
                    &tensor.data_type,
                ),
                GGMLType::BF16 => GGUFTensor::from_raw_parts::<bf16>(
                    bytes,
                    size,
                    &tensor.shape,
                    &tensor.data_type,
                ),
                GGMLType::F16 => {
                    GGUFTensor::from_raw_parts::<f16>(bytes, size, &tensor.shape, &tensor.data_type)
                }
                GGMLType::F32 => {
                    GGUFTensor::from_raw_parts::<f32>(bytes, size, &tensor.shape, &tensor.data_type)
                }
                _ => unimplemented!(),
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
    pub fn block_size(&self) -> usize {
        match self {
            Self::Q4_0 => std::mem::size_of::<crate::BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<crate::BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<crate::BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<crate::BlockQ5_1>(),
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Self::Q8_0 => std::mem::size_of::<crate::BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<crate::BlockQ8_1>(),
            Self::Q2_K => std::mem::size_of::<crate::BlockQ2K>(),
            Self::Q3_K => std::mem::size_of::<crate::BlockQ3K>(),
            Self::Q4_K => std::mem::size_of::<crate::BlockQ4K>(),
            Self::Q5_K => std::mem::size_of::<crate::BlockQ5K>(),
            Self::Q6_K => std::mem::size_of::<crate::BlockQ6K>(),
            Self::Q8_K => std::mem::size_of::<crate::BlockQ8K>(),
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_2 => todo!(),
            Self::Q4_3 => todo!(),
            Self::IQ2_XXS => std::mem::size_of::<crate::BlockIq2XXS>(),
            Self::IQ2_XS => std::mem::size_of::<crate::BlockIq2XS>(),
            Self::IQ3_XXS => std::mem::size_of::<crate::BlockIq3XXS>(),
            Self::IQ1_S => std::mem::size_of::<crate::BlockIq1S>(),
            Self::IQ4_NL => std::mem::size_of::<crate::BlockIq4NL>(),
            Self::IQ3_S => std::mem::size_of::<crate::BlockIq3S>(),
            Self::IQ2_S => std::mem::size_of::<crate::BlockIq2S>(),
            Self::IQ4_XS => std::mem::size_of::<crate::BlockIq4XS>(),
            Self::I8 => todo!(),
            Self::I16 => todo!(),
            Self::I32 => todo!(),
            Self::I64 => todo!(),
            Self::F64 => todo!(),
            Self::IQ1_M => std::mem::size_of::<crate::BlockIq1M>(),
            Self::BF16 => 2,
            Self::Q4_0_4_4 => todo!(),
            Self::Q4_0_4_8 => todo!(),
            Self::Q4_0_8_8 => todo!(),
            Self::TQ1_0 => std::mem::size_of::<crate::BlockTq1_0>(),
            Self::TQ2_0 => std::mem::size_of::<crate::BlockTq2_0>(),
        }
    }
    pub fn size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::Q4_0 => crate::QK4_0,
            Self::Q4_1 => crate::QK4_1,
            Self::Q5_0 => crate::QK5_0,
            Self::Q5_1 => crate::QK5_1,
            Self::Q8_0 => crate::QK8_0,
            Self::Q8_1 => crate::QK8_1,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => {
                crate::QK_K
            }
            _ => unimplemented!(),
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
