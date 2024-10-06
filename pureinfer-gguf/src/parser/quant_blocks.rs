use half::{bf16, f16, slice::HalfFloatSliceExt};

pub trait GGUFBlock: Send + Sync + Sized + Clone {
    const DTYPE: crate::GGMLType = crate::GGMLType::F32;
    const BLCK_SIZE: usize = 1;
    #[allow(unused_variables)]
    fn to_f32(s: &[Self], f: &mut [f32]) -> crate::Result<()> {
        unimplemented!()
    }
    #[allow(unused_variables)]
    fn from_f32(f: &[f32], s: &mut [Self]) -> crate::Result<()> {
        unimplemented!()
    }
    fn zeros() -> Self {
        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }
    }
}

pub const QK4_0: usize = 32;

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ4_0 {
    d: f16,              // delta
    qs: [u8; QK4_0 / 2], // nibbles / quants
}

const _: () = assert!(
    std::mem::size_of::<BlockQ4_0>() == std::mem::size_of::<f16>() + QK4_0 / 2,
    "wrong q4_0 block size/padding"
);

impl GGUFBlock for BlockQ4_0 {
    const DTYPE: crate::GGMLType = crate::GGMLType::Q4_0;

    const BLCK_SIZE: usize = QK4_0;
    fn to_f32(s: &[Self], f: &mut [f32]) -> crate::Result<()> {
        let k = f.len();
        let qk = Self::BLCK_SIZE;
        if k % qk != 0 {
            return Err(crate::Error::QuantizationError(format!(
                "dequantize_row_q4_0: {k} is not divisible by {qk}"
            )));
        }

        let nb = k / qk;
        for i in 0..nb {
            let d = s[i].d.to_f32();

            for j in 0..(qk / 2) {
                let x0 = (s[i].qs[j] & 0x0F) as i16 - 8;
                let x1 = (s[i].qs[j] >> 4) as i16 - 8;

                f[i * qk + j] = (x0 as f32) * d;
                f[i * qk + j + qk / 2] = (x1 as f32) * d;
            }
        }
        Ok(())
    }

    fn from_f32(f: &[f32], s: &mut [Self]) -> crate::Result<()> {
        let qk = Self::BLCK_SIZE;
        let k = s.len();
        if k % qk != 0 {
            return Err(crate::Error::QuantizationError(format!(
                "{k} is not divisible by {qk}"
            )));
        };
        let nb = k / qk;
        if f.len() != nb {
            return Err(crate::Error::QuantizationError(format!(
                "size mismatch {} {} {}",
                s.len(),
                f.len(),
                qk,
            )));
        }
        for (i, ys) in s.iter_mut().enumerate() {
            let mut amax = 0f32;
            let mut max = 0f32;

            let xs = &f[i * qk..(i + 1) * qk];
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            let d = max / -8.0;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);

            for (j, q) in ys.qs.iter_mut().enumerate() {
                let x0 = xs[j] * id;
                let x1 = xs[qk / 2 + j] * id;
                let xi0 = u8::min(15, (x0 + 8.5) as u8);
                let xi1 = u8::min(15, (x1 + 8.5) as u8);
                *q = xi0 | (xi1 << 4)
            }
        }
        Ok(())
    }
}

pub const QK4_1: usize = 32;
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ4_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qs: [u8; QK4_1 / 2],
}
const _: () = assert!(
    std::mem::size_of::<BlockQ4_1>() == 2 * std::mem::size_of::<f16>() + QK4_1 / 2,
    "wrong q4_1 block size/padding"
);
impl GGUFBlock for BlockQ4_1 {}

pub const QK5_0: usize = 32;

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ5_0 {
    d: f16,              // delta
    qh: [u8; 4],         // 5-th bit of quants
    qs: [u8; QK5_0 / 2], // nibbles / quants
}

const _: () = assert!(
    std::mem::size_of::<BlockQ5_0>() == std::mem::size_of::<f16>() + 4 + QK5_0 / 2,
    "wrong q5_0 block size/padding"
);
impl GGUFBlock for BlockQ5_0 {}

pub const QK5_1: usize = 32;

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ5_1 {
    pub dm: [f16; 2],        // delta and min
    pub qh: [u8; 4],         // 5-th bit of quants
    pub qs: [u8; QK5_1 / 2], // nibbles / quants
}

const _: () = assert!(
    std::mem::size_of::<BlockQ5_1>() == 2 * std::mem::size_of::<f16>() + 4 + QK5_1 / 2,
    "wrong q5_1 block size/padding"
);
impl GGUFBlock for BlockQ5_1 {}

pub const QK8_0: usize = 32;

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ8_0 {
    pub d: f16,          // delta
    pub qs: [i8; QK8_0], // quants
}

const _: () = assert!(
    std::mem::size_of::<BlockQ8_0>() == std::mem::size_of::<f16>() + QK8_0,
    "wrong q8_0 block size/padding"
);
impl GGUFBlock for BlockQ8_0 {}

pub const QK8_1: usize = 32;

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ8_1 {
    pub ds: [f16; 2],    // delta and d * sum(qs[i])
    pub qs: [i8; QK8_1], // quants
}

const _: () = assert!(
    std::mem::size_of::<BlockQ8_1>() == 2 * std::mem::size_of::<f16>() + QK8_1,
    "wrong q8_1 block size/padding"
);
impl GGUFBlock for BlockQ8_1 {}

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ4_0x4 {
    pub d: [f16; 4],         // deltas for 4 q4_0 blocks
    pub qs: [u8; QK4_0 * 2], // nibbles / quants for 4 q4_0 blocks
}

const _: () = assert!(
    std::mem::size_of::<BlockQ4_0x4>() == 4 * std::mem::size_of::<f16>() + QK4_0 * 2,
    "wrong q4_0x4 block size/padding"
);
impl GGUFBlock for BlockQ4_0x4 {}

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ4_0x8 {
    pub d: [f16; 8],         // deltas for 8 q4_0 blocks
    pub qs: [u8; QK4_0 * 4], // nibbles / quants for 8 q4_0 blocks
}

const _: () = assert!(
    std::mem::size_of::<BlockQ4_0x8>() == 8 * std::mem::size_of::<f16>() + QK4_0 * 4,
    "wrong q4_0x8 block size/padding"
);
impl GGUFBlock for BlockQ4_0x8 {}

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ8_0x4 {
    pub d: [f16; 4],         // deltas for 4 q8_0 blocks
    pub qs: [i8; QK8_0 * 4], // quants for 4 q8_0 blocks
}

const _: () = assert!(
    std::mem::size_of::<BlockQ8_0x4>() == 4 * std::mem::size_of::<f16>() + QK8_0 * 4,
    "wrong q8_0x4 block size/padding"
);
impl GGUFBlock for BlockQ8_0x4 {}

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ8_0x8 {
    pub d: [f16; 8],         // deltas for 8 q8_0 blocks
    pub qs: [i8; QK8_0 * 8], // quants for 8 q8_0 blocks
}

const _: () = assert!(
    std::mem::size_of::<BlockQ8_0x8>() == 8 * std::mem::size_of::<f16>() + QK8_0 * 8,
    "wrong q8_0x8 block size/padding"
);
impl GGUFBlock for BlockQ8_0x8 {}

pub const QK_K: usize = 256;

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockTq1_0 {
    pub qs: [u8; (QK_K - 4 * QK_K / 64) / 5], // 5 elements per byte (3^5 = 243 < 256)
    pub qh: [u8; QK_K / 64],                  // 4 elements per byte
    pub d: f16,                               // delta
}

const _: () = assert!(
    std::mem::size_of::<BlockTq1_0>()
        == std::mem::size_of::<f16>() + QK_K / 64 + (QK_K - 4 * QK_K / 64) / 5,
    "wrong tq1_0 block size/padding"
);
impl GGUFBlock for BlockTq1_0 {}

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockTq2_0 {
    pub qs: [u8; QK_K / 4], // 2 bits per element
    pub d: f16,             // delta
}

const _: () = assert!(
    std::mem::size_of::<BlockTq2_0>() == std::mem::size_of::<f16>() + QK_K / 4,
    "wrong tq2_0 block size/padding"
);
impl GGUFBlock for BlockTq2_0 {}

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ2K {
    pub scales: [u8; QK_K / 16], // scales and mins, quantized with 4 bits
    pub qs: [u8; QK_K / 4],      // quants
    pub dm: [f16; 2],            // super-block scale for quantized scales and mins
}

const _: () = assert!(
    std::mem::size_of::<BlockQ2K>() == 2 * std::mem::size_of::<f16>() + QK_K / 16 + QK_K / 4,
    "wrong q2_K block size/padding"
);
impl GGUFBlock for BlockQ2K {
    const DTYPE: crate::GGMLType = crate::GGMLType::Q2_K;

    const BLCK_SIZE: usize = QK_K;
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ3K {
    d: f16,           // super-block scale
    scales: [u8; 12], // quantized scales
    hmask: [u8; QK_K / 4],
    qs: [u8; QK_K / 8], // low 2 bits of quants
}

const _: () = assert!(
    std::mem::size_of::<BlockQ3K>() == std::mem::size_of::<f16>() + QK_K / 4 + QK_K / 8 + 12,
    "wrong q3_K block size/padding"
);
impl GGUFBlock for BlockQ3K {
    const DTYPE: crate::GGMLType = crate::GGMLType::Q3_K;

    const BLCK_SIZE: usize = QK_K;
}
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ4K {
    d: f16,                     // super-block scale for quantized scales
    dmin: f16,                  // super-block scale for quantized mins
    scales: [u8; K_SCALE_SIZE], // quantized scales and mins
    qs: [u8; QK_K / 2],         // 4-bit quants
}

const _: () = assert!(
    std::mem::size_of::<BlockQ4K>() == 2 * std::mem::size_of::<f16>() + K_SCALE_SIZE + QK_K / 2,
    "wrong q4_K block size/padding"
);

impl GGUFBlock for BlockQ4K {
    const DTYPE: crate::GGMLType = crate::GGMLType::Q4_K;

    const BLCK_SIZE: usize = QK_K;
    fn to_f32(s: &[Self], f: &mut [f32]) -> crate::Result<()> {
        for (block, y) in crate::group_for_dequantization(s, f)? {
            let d = block.d.to_f32();
            let min = block.dmin.to_f32();
            let q = &block.qs;
            let mut is = 0;
            let mut ys_index = 0;

            for j in (0..QK_K).step_by(64) {
                let q = &q[j / 2..j / 2 + 32];
                let (sc, m) = crate::get_scale_min_k4(is, &block.scales);
                let d1 = d * sc as f32;
                let m1 = min * m as f32;
                let (sc, m) = crate::get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * sc as f32;
                let m2 = min * m as f32;
                for q in q {
                    y[ys_index] = d1 * (q & 0xF) as f32 - m1;
                    ys_index += 1;
                }
                for q in q {
                    y[ys_index] = d2 * (q >> 4) as f32 - m2;
                    ys_index += 1;
                }
                is += 2;
            }
        }
        Ok(())
    }
    fn from_f32(f: &[f32], s: &mut [Self]) -> crate::Result<()> {
        for (block, x) in crate::group_for_quantization(f, s)? {
            let mut mins: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut scales: [f32; QK_K / 32] = [0.0; QK_K / 32];

            for (j, x_scale_slice) in x.chunks_exact(32).enumerate() {
                (scales[j], mins[j]) = crate::make_qkx1_quants(15, 5, x_scale_slice);
            }

            // get max scale and max min and ensure they are >= 0.0
            let max_scale = scales.iter().fold(0.0, |max, &val| val.max(max));
            let max_min = mins.iter().fold(0.0, |max, &val| val.max(max));

            let inv_scale = if max_scale > 0.0 {
                63.0 / max_scale
            } else {
                0.0
            };
            let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

            for j in 0..QK_K / 32 {
                let ls = crate::nearest_int(inv_scale * scales[j]).min(63) as u8;
                let lm = crate::nearest_int(inv_min * mins[j]).min(63) as u8;
                if j < 4 {
                    block.scales[j] = ls;
                    block.scales[j + 4] = lm;
                } else {
                    block.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                    block.scales[j - 4] |= (ls >> 4) << 6;
                    block.scales[j] |= (lm >> 4) << 6;
                }
            }

            block.d = f16::from_f32(max_scale / 63.0);
            block.dmin = f16::from_f32(max_min / 63.0);

            let mut l: [u8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 32 {
                let (sc, m) = crate::get_scale_min_k4(j, &block.scales);
                let d = block.d.to_f32() * sc as f32;
                if d != 0.0 {
                    let dm = block.dmin.to_f32() * m as f32;
                    for ii in 0..32 {
                        let l_val = crate::nearest_int((x[32 * j + ii] + dm) / d);
                        l[32 * j + ii] = l_val.clamp(0, 15) as u8;
                    }
                }
            }

            let q = &mut block.qs;
            for j in (0..QK_K).step_by(64) {
                for l_val in 0..32 {
                    let offset_index = (j / 64) * 32 + l_val;
                    q[offset_index] = l[j + l_val] | (l[j + l_val + 32] << 4);
                }
            }
        }
        Ok(())
    }
}

pub const K_SCALE_SIZE: usize = 12;
pub const QK5_K: usize = 64;
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ5K {
    d: f16,                     // super-block scale for quantized scales
    dmin: f16,                  // super-block scale for quantized mins
    scales: [u8; K_SCALE_SIZE], // quantized scales and mins
    qh: [u8; QK5_K / 8],        // quants, high bit
    qs: [u8; QK5_K / 2],        // quants, low 4 bits
}

const _: () = assert!(
    std::mem::size_of::<BlockQ5K>()
        == 2 * std::mem::size_of::<f16>() + K_SCALE_SIZE + QK5_K / 2 + QK5_K / 8,
    "wrong q5_K block size/padding"
);
impl GGUFBlock for BlockQ5K {
    const DTYPE: crate::GGMLType = crate::GGMLType::Q5_K;

    const BLCK_SIZE: usize = QK_K;
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ6K {
    ql: [u8; QK_K / 2],      // quants, lower 4 bits
    qh: [u8; QK_K / 4],      // quants, upper 2 bits
    scales: [i8; QK_K / 16], // quantized scales
    d: f16,                  // super-block scale
}

const _: () = assert!(
    std::mem::size_of::<BlockQ6K>() == std::mem::size_of::<f16>() + QK_K / 16 + 3 * QK_K / 4,
    "wrong q6_K block size/padding"
);

impl GGUFBlock for BlockQ6K {
    const DTYPE: crate::GGMLType = crate::GGMLType::Q6_K;

    const BLCK_SIZE: usize = QK_K;
    fn to_f32(s: &[Self], f: &mut [f32]) -> crate::Result<()> {
        let k = f.len();
        if k % QK_K != 0 {
            return Err(crate::Error::QuantizationError(format!(
                "dequantize_row_q6k: {k} is not divisible by {QK_K}"
            )));
        }
        for (idx_x, x) in s.iter().enumerate() {
            let d = x.d.to_f32();
            let ql = &x.ql;
            let qh = &x.qh;
            let sc = &x.scales;
            for n in (0..QK_K).step_by(128) {
                let idx = n / 128;
                let ys = &mut f[idx_x * QK_K + n..];
                let sc = &sc[8 * idx..];
                let ql = &ql[64 * idx..];
                let qh = &qh[32 * idx..];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                    ys[l] = d * sc[is] as f32 * q1 as f32;
                    ys[l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                    ys[l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                    ys[l + 96] = d * sc[is + 6] as f32 * q4 as f32;
                }
            }
        }
        Ok(())
    }
    fn from_f32(f: &[f32], s: &mut [Self]) -> crate::Result<()> {
        if s.len() != f.len() * Self::BLCK_SIZE {
            return Err(crate::Error::QuantizationError(format!(
                "quantize_row_q6k: size mismatch {} {} {}",
                s.len(),
                f.len(),
                Self::BLCK_SIZE
            )));
        }
        let mut l = [0i8; QK_K];
        let mut scales = [0f32; QK_K / 16];
        let mut x = f.as_ptr();
        let l = l.as_mut_ptr();
        unsafe {
            for y in s.iter_mut() {
                let mut max_scale = 0f32;
                let mut max_abs_scale = 0f32;
                for (ib, scale_) in scales.iter_mut().enumerate() {
                    let scale = crate::make_qx_quants(16, 32, x.add(16 * ib), l.add(16 * ib), 1);
                    *scale_ = scale;
                    let abs_scale = scale.abs();
                    if abs_scale > max_abs_scale {
                        max_abs_scale = abs_scale;
                        max_scale = scale
                    }
                }

                let iscale = -128f32 / max_scale;
                y.d = f16::from_f32(1.0 / iscale);

                for (y_scale, scale) in y.scales.iter_mut().zip(scales.iter()) {
                    *y_scale = crate::nearest_int(iscale * scale).min(127) as i8
                }

                for (j, &y_scale) in y.scales.iter().enumerate() {
                    let d = y.d.to_f32() * y_scale as f32;
                    if d == 0. {
                        continue;
                    }
                    for ii in 0..16 {
                        let ll = crate::nearest_int(*x.add(16 * j + ii) / d).clamp(-32, 31);
                        *l.add(16 * j + ii) = (ll + 32) as i8
                    }
                }

                let mut ql = y.ql.as_mut_ptr();
                let mut qh = y.qh.as_mut_ptr();

                for j in (0..QK_K).step_by(128) {
                    for l_idx in 0..32 {
                        let q1 = *l.add(j + l_idx) & 0xF;
                        let q2 = *l.add(j + l_idx + 32) & 0xF;
                        let q3 = *l.add(j + l_idx + 64) & 0xF;
                        let q4 = *l.add(j + l_idx + 96) & 0xF;
                        *ql.add(l_idx) = (q1 | (q3 << 4)) as u8;
                        *ql.add(l_idx + 32) = (q2 | (q4 << 4)) as u8;
                        *qh.add(l_idx) = ((*l.add(j + l_idx) >> 4)
                            | ((*l.add(j + l_idx + 32) >> 4) << 2)
                            | ((*l.add(j + l_idx + 64) >> 4) << 4)
                            | ((*l.add(j + l_idx + 96) >> 4) << 6))
                            as u8;
                    }
                    ql = ql.add(64);
                    qh = qh.add(32);
                }

                x = x.add(QK_K)
            }
        }
        Ok(())
    }
}

pub const QK8_K: usize = 64;
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ8K {
    d: f32,                   // delta
    qs: [i8; QK8_K],          // quants
    bsums: [i16; QK8_K / 16], // sum of quants in groups of 16
}

const _: () = assert!(
    std::mem::size_of::<BlockQ8K>()
        == std::mem::size_of::<f32>() + QK8_K + QK8_K / 16 * std::mem::size_of::<i16>(),
    "wrong q8_K block size/padding"
);
impl GGUFBlock for BlockQ8K {
    const DTYPE: crate::GGMLType = crate::GGMLType::Q8_K;

    const BLCK_SIZE: usize = QK_K;
}

pub const QK2_XXS_K: usize = 64;
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockIq2XXS {
    d: f16,                   // delta
    qs: [u16; QK2_XXS_K / 8], // quants
}

const _: () = assert!(
    std::mem::size_of::<BlockIq2XXS>()
        == std::mem::size_of::<f16>() + QK2_XXS_K / 8 * std::mem::size_of::<u16>(),
    "wrong iq2_xxs block size/padding"
);

impl GGUFBlock for BlockIq2XXS {}
// 2.3125 bpw quants
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockIq2XS {
    d: f16,
    qs: [u16; QK_K / 8],
    scales: [u8; QK_K / 32],
}

const _: () = assert!(
    std::mem::size_of::<BlockIq2XS>()
        == std::mem::size_of::<f16>() + QK_K / 8 * std::mem::size_of::<u16>() + QK_K / 32,
    "wrong iq2_xs block size/padding"
);
impl GGUFBlock for BlockIq2XS {}

// 2.5625 bpw quants
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockIq2S {
    d: f16,
    qs: [u8; QK_K / 4],
    qh: [u8; QK_K / 32],
    scales: [u8; QK_K / 32],
}

const _: () = assert!(
    std::mem::size_of::<BlockIq2S>() == std::mem::size_of::<f16>() + QK_K / 4 + QK_K / 16,
    "wrong iq2_s block size/padding"
);
impl GGUFBlock for BlockIq2S {}

// 3.0625 bpw quants
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockIq3XXS {
    d: f16,
    qs: [u8; 3 * QK_K / 8],
}

const _: () = assert!(
    std::mem::size_of::<BlockIq3XXS>() == std::mem::size_of::<f16>() + 3 * (QK_K / 8),
    "wrong iq3_xxs block size/padding"
);
impl GGUFBlock for BlockIq3XXS {}

// 3.4375 bpw
const IQ3S_N_SCALE: usize = QK_K / 64;
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockIq3S {
    d: f16,
    qs: [u8; QK_K / 4],
    qh: [u8; QK_K / 32],
    signs: [u8; QK_K / 8],
    scales: [u8; IQ3S_N_SCALE],
}

const _: () = assert!(
    std::mem::size_of::<BlockIq3S>()
        == std::mem::size_of::<f16>() + 13 * (QK_K / 32) + IQ3S_N_SCALE,
    "wrong iq3_s block size/padding"
);
impl GGUFBlock for BlockIq3S {}

// 1.5625 bpw
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockIq1S {
    d: f16,
    qs: [u8; QK_K / 8],
    qh: [u16; QK_K / 32],
}

const _: () = assert!(
    std::mem::size_of::<BlockIq1S>() == std::mem::size_of::<f16>() + QK_K / 8 + QK_K / 16,
    "wrong iq1_s block size/padding"
);
impl GGUFBlock for BlockIq1S {}

// 1.75 bpw
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockIq1M {
    qs: [u8; QK_K / 8],
    qh: [u8; QK_K / 16],
    scales: [u8; QK_K / 32],
}

const _: () = assert!(
    std::mem::size_of::<BlockIq1M>() == QK_K / 8 + QK_K / 16 + QK_K / 32,
    "wrong iq1_m block size/padding"
);
impl GGUFBlock for BlockIq1M {}
// Non-linear quants
pub const QK4_NL: usize = 32;
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockIq4NL {
    d: f16,
    qs: [u8; QK4_NL / 2],
}

const _: () = assert!(
    std::mem::size_of::<BlockIq4NL>() == std::mem::size_of::<f16>() + QK4_NL / 2,
    "wrong iq4_nl block size/padding"
);
impl GGUFBlock for BlockIq4NL {}

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockIq4XS {
    d: f16,
    scales_h: u16,
    scales_l: [u8; QK_K / 64],
    qs: [u8; QK_K / 2],
}

const _: () = assert!(
    std::mem::size_of::<BlockIq4XS>()
        == std::mem::size_of::<f16>() + std::mem::size_of::<u16>() + QK_K / 64 + QK_K / 2,
    "wrong iq4_xs block size/padding"
);
impl GGUFBlock for BlockIq4XS {}

impl GGUFBlock for f16 {
    const DTYPE: crate::GGMLType = crate::GGMLType::F16;
    const BLCK_SIZE: usize = 1;
    fn to_f32(s: &[Self], f: &mut [f32]) -> crate::Result<()> {
        s.convert_to_f32_slice(f);
        Ok(())
    }
    fn from_f32(f: &[f32], s: &mut [Self]) -> crate::Result<()> {
        s.convert_from_f32_slice(f);
        Ok(())
    }
}

impl GGUFBlock for bf16 {
    const DTYPE: crate::GGMLType = crate::GGMLType::BF16;
    const BLCK_SIZE: usize = 1;
    fn to_f32(s: &[Self], f: &mut [f32]) -> crate::Result<()> {
        s.convert_to_f32_slice(f);
        Ok(())
    }
    fn from_f32(f: &[f32], s: &mut [Self]) -> crate::Result<()> {
        s.convert_from_f32_slice(f);
        Ok(())
    }
}

impl GGUFBlock for f32 {
    const DTYPE: crate::GGMLType = crate::GGMLType::F32;
    const BLCK_SIZE: usize = 1;
    #[allow(unused_variables)]
    fn to_f32(s: &[Self], f: &mut [f32]) -> crate::Result<()> {
        f.copy_from_slice(s);
        Ok(())
    }
    #[allow(unused_variables)]
    fn from_f32(f: &[f32], s: &mut [Self]) -> crate::Result<()> {
        s.copy_from_slice(f);
        Ok(())
    }
}

pub trait BaseGGUFBlock: Send + Sync {
    fn to_f32(&self, elem: usize) -> crate::Result<Vec<f32>>;
    fn from_f32(f: &[f32], elem: usize) -> crate::Result<Self>
    where
        Self: Sized;
    fn len(&self) -> usize;
}
impl<T> BaseGGUFBlock for Vec<T>
where
    T: GGUFBlock,
{
    fn to_f32(&self, elem: usize) -> crate::Result<Vec<f32>> {
        let mut f = vec![0.0; elem];
        T::to_f32(self, &mut f)?;
        Ok(f)
    }
    fn from_f32(f: &[f32], elem: usize) -> crate::Result<Self> {
        let mut s = vec![T::zeros(); elem];
        T::from_f32(f, &mut s)?;
        Ok(s)
    }
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T> BaseGGUFBlock for &[T]
where
    T: GGUFBlock,
{
    fn to_f32(&self, elem: usize) -> crate::Result<Vec<f32>> {
        let mut f = vec![0.0; elem];
        T::to_f32(self, &mut f)?;
        Ok(f)
    }
    #[allow(unused_variables)]
    fn from_f32(f: &[f32], elem: usize) -> crate::Result<Self> {
        Err(crate::Error::QuantizationError(
            "How are you calling this stupid function?".to_string(),
        ))
    }
    fn len(&self) -> usize {
        self.to_vec().len()
    }
}
