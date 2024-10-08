use safetensors::tensor::Metadata;
use std::io::Read;

pub struct SafeTensors<R>
where
    R: Read,
{
    pub header: Metadata,
    pub reader: R,
    pub offset: usize,
}

pub struct STParser<R>
where
    R: Read,
{
    bytes: R,
    offset: usize,
}

impl<R> STParser<R>
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
    pub fn parse(mut self) -> crate::Result<SafeTensors<R>> {
        let len = u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap());
        let header = self.read_bytes(len as usize)?;
        let header: Metadata = serde_json::from_slice(&header)?;
        Ok(SafeTensors {
            header,
            reader: self.bytes,
            offset: self.offset,
        })
    }
}
