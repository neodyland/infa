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
    COUNT,
}
