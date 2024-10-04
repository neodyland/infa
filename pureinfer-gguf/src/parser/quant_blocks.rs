use half::f16;

pub const QK4_0: usize = 32;

#[repr(C)]
pub struct BlockQ4_0 {
    d: f16,              // delta
    qs: [u8; QK4_0 / 2], // nibbles / quants
}

const _: () = assert!(
    std::mem::size_of::<BlockQ4_0>() == std::mem::size_of::<f16>() + QK4_0 / 2,
    "wrong q4_0 block size/padding"
);

pub const QK4_1: usize = 32;
#[repr(C)]
pub struct BlockQ4_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qs: [u8; QK4_1 / 2],
}
const _: () = assert!(
    std::mem::size_of::<BlockQ4_1>() == 2 * std::mem::size_of::<f16>() + QK4_1 / 2,
    "wrong q4_1 block size/padding"
);

pub const QK5_0: usize = 32;

#[repr(C)]
pub struct BlockQ5_0 {
    d: f16,              // delta
    qh: [u8; 4],         // 5-th bit of quants
    qs: [u8; QK5_0 / 2], // nibbles / quants
}

const _: () = assert!(
    std::mem::size_of::<BlockQ5_0>() == std::mem::size_of::<f16>() + 4 + QK5_0 / 2,
    "wrong q5_0 block size/padding"
);

pub const QK5_1: usize = 32;

#[repr(C)]
pub struct BlockQ5_1 {
    pub dm: [f16; 2],        // delta and min
    pub qh: [u8; 4],         // 5-th bit of quants
    pub qs: [u8; QK5_1 / 2], // nibbles / quants
}

const _: () = assert!(
    std::mem::size_of::<BlockQ5_1>() == 2 * std::mem::size_of::<f16>() + 4 + QK5_1 / 2,
    "wrong q5_1 block size/padding"
);

pub const QK8_0: usize = 32;

#[repr(C)]
pub struct BlockQ8_0 {
    pub d: f16,          // delta
    pub qs: [i8; QK8_0], // quants
}

const _: () = assert!(
    std::mem::size_of::<BlockQ8_0>() == std::mem::size_of::<f16>() + QK8_0,
    "wrong q8_0 block size/padding"
);

pub const QK8_1: usize = 32;

#[repr(C)]
pub struct BlockQ8_1 {
    pub ds: [f16; 2],    // delta and d * sum(qs[i])
    pub qs: [i8; QK8_1], // quants
}

const _: () = assert!(
    std::mem::size_of::<BlockQ8_1>() == 2 * std::mem::size_of::<f16>() + QK8_1,
    "wrong q8_1 block size/padding"
);

#[repr(C)]
pub struct BlockQ4_0x4 {
    pub d: [f16; 4],         // deltas for 4 q4_0 blocks
    pub qs: [u8; QK4_0 * 2], // nibbles / quants for 4 q4_0 blocks
}

const _: () = assert!(
    std::mem::size_of::<BlockQ4_0x4>() == 4 * std::mem::size_of::<f16>() + QK4_0 * 2,
    "wrong q4_0x4 block size/padding"
);

#[repr(C)]
pub struct BlockQ4_0x8 {
    pub d: [f16; 8],         // deltas for 8 q4_0 blocks
    pub qs: [u8; QK4_0 * 4], // nibbles / quants for 8 q4_0 blocks
}

const _: () = assert!(
    std::mem::size_of::<BlockQ4_0x8>() == 8 * std::mem::size_of::<f16>() + QK4_0 * 4,
    "wrong q4_0x8 block size/padding"
);

#[repr(C)]
pub struct BlockQ8_0x4 {
    pub d: [f16; 4],         // deltas for 4 q8_0 blocks
    pub qs: [i8; QK8_0 * 4], // quants for 4 q8_0 blocks
}

const _: () = assert!(
    std::mem::size_of::<BlockQ8_0x4>() == 4 * std::mem::size_of::<f16>() + QK8_0 * 4,
    "wrong q8_0x4 block size/padding"
);

#[repr(C)]
pub struct BlockQ8_0x8 {
    pub d: [f16; 8],         // deltas for 8 q8_0 blocks
    pub qs: [i8; QK8_0 * 8], // quants for 8 q8_0 blocks
}

const _: () = assert!(
    std::mem::size_of::<BlockQ8_0x8>() == 8 * std::mem::size_of::<f16>() + QK8_0 * 8,
    "wrong q8_0x8 block size/padding"
);

pub const QK_K: usize = 256;

#[repr(C)]
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

#[repr(C)]
pub struct BlockTq2_0 {
    pub qs: [u8; QK_K / 4], // 2 bits per element
    pub d: f16,             // delta
}

const _: () = assert!(
    std::mem::size_of::<BlockTq2_0>() == std::mem::size_of::<f16>() + QK_K / 4,
    "wrong tq2_0 block size/padding"
);

#[repr(C)]
pub struct BlockQ2K {
    pub scales: [u8; QK_K / 16], // scales and mins, quantized with 4 bits
    pub qs: [u8; QK_K / 4],      // quants
    pub dm: [f16; 2],            // super-block scale for quantized scales and mins
}

const _: () = assert!(
    std::mem::size_of::<BlockQ2K>() == 2 * std::mem::size_of::<f16>() + QK_K / 16 + QK_K / 4,
    "wrong q2_K block size/padding"
);

#[repr(C)]
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

pub const QK4_K: usize = 64;
#[repr(C)]
pub struct BlockQ4K {
    d: f16,                     // super-block scale for quantized scales
    dmin: f16,                  // super-block scale for quantized mins
    scales: [u8; K_SCALE_SIZE], // quantized scales and mins
    qs: [u8; QK4_K / 2],        // 4-bit quants
}

const _: () = assert!(
    std::mem::size_of::<BlockQ4K>() == 2 * std::mem::size_of::<f16>() + K_SCALE_SIZE + QK4_K / 2,
    "wrong q4_K block size/padding"
);

pub const K_SCALE_SIZE: usize = 12;
pub const QK5_K: usize = 64;
#[repr(C)]
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

pub const QK6_K: usize = 64;
#[repr(C)]
pub struct BlockQ6K {
    ql: [u8; QK6_K / 2],      // quants, lower 4 bits
    qh: [u8; QK6_K / 4],      // quants, upper 2 bits
    scales: [i8; QK6_K / 16], // quantized scales
    d: f16,                   // super-block scale
}

const _: () = assert!(
    std::mem::size_of::<BlockQ6K>() == std::mem::size_of::<f16>() + QK6_K / 16 + 3 * QK6_K / 4,
    "wrong q6_K block size/padding"
);

pub const QK8_K: usize = 64;
#[repr(C)]
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

pub const QK2_XXS_K: usize = 64;
#[repr(C)]
pub struct BlockIq2XXS {
    d: f16,                   // delta
    qs: [u16; QK2_XXS_K / 8], // quants
}

const _: () = assert!(
    std::mem::size_of::<BlockIq2XXS>()
        == std::mem::size_of::<f16>() + QK2_XXS_K / 8 * std::mem::size_of::<u16>(),
    "wrong iq2_xxs block size/padding"
);

// 2.3125 bpw quants
#[repr(C)]
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

// 2.5625 bpw quants
#[repr(C)]
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

// 3.0625 bpw quants
#[repr(C)]
pub struct BlockIq3XXS {
    d: f16,
    qs: [u8; 3 * QK_K / 8],
}

const _: () = assert!(
    std::mem::size_of::<BlockIq3XXS>() == std::mem::size_of::<f16>() + 3 * (QK_K / 8),
    "wrong iq3_xxs block size/padding"
);

// 3.4375 bpw
const IQ3S_N_SCALE: usize = QK_K / 64;
#[repr(C)]
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

// 1.5625 bpw
#[repr(C)]
pub struct BlockIq1S {
    d: f16,
    qs: [u8; QK_K / 8],
    qh: [u16; QK_K / 32],
}

const _: () = assert!(
    std::mem::size_of::<BlockIq1S>() == std::mem::size_of::<f16>() + QK_K / 8 + QK_K / 16,
    "wrong iq1_s block size/padding"
);

// 1.75 bpw
#[repr(C)]
pub struct BlockIq1M {
    qs: [u8; QK_K / 8],
    qh: [u8; QK_K / 16],
    scales: [u8; QK_K / 32],
}

const _: () = assert!(
    std::mem::size_of::<BlockIq1M>() == QK_K / 8 + QK_K / 16 + QK_K / 32,
    "wrong iq1_m block size/padding"
);

// Used by IQ1_M quants
#[repr(C)]
pub union Iq1mScale {
    f16: f16,
    u16: u16,
}

// Non-linear quants
pub const QK4_NL: usize = 32;
#[repr(C)]
pub struct BlockIq4NL {
    d: f16,
    qs: [u8; QK4_NL / 2],
}

const _: () = assert!(
    std::mem::size_of::<BlockIq4NL>() == std::mem::size_of::<f16>() + QK4_NL / 2,
    "wrong iq4_nl block size/padding"
);

#[repr(C)]
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
