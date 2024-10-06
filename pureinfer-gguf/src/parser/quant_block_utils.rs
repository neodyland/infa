pub fn nearest_int(v: f32) -> i32 {
    v.round() as i32
}

pub fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

pub fn group_for_quantization<'a, 'b, T: crate::GGUFBlock>(
    xs: &'b [f32],
    ys: &'a mut [T],
) -> crate::Result<Vec<(&'a mut T, &'b [f32])>> {
    let block_size = T::BLCK_SIZE;
    let dtype = T::DTYPE;

    let expected_blocks = xs.len() / block_size;
    let actual_blocks = ys.len();

    // Validate that the input is the right size
    if expected_blocks != actual_blocks {
        return Err(crate::Error::QuantizationError(format!("quantize {dtype:?}: expected {expected_blocks} blocks but only {actual_blocks} were provided!")));
    }

    Ok(ys.iter_mut().zip(xs.chunks_exact(block_size)).collect())
}

/// Validates that the input and output are the right size and returns an iterator which maps each
/// input block `xs` to its corresponding output region in `ys`. Each output region is guaranteed
/// to be `T::BLCK_SIZE` long.
pub fn group_for_dequantization<'a, 'b, T: crate::GGUFBlock>(
    xs: &'a [T],
    ys: &'b mut [f32],
) -> crate::Result<Vec<(&'a T, &'b mut [f32])>> {
    let block_size = T::BLCK_SIZE;
    let dtype = T::DTYPE;

    let actual_output_len = ys.len();
    let expected_output_len = xs.len() * block_size;
    // Validate that the output is the right size
    if expected_output_len != actual_output_len {
        return Err(crate::Error::QuantizationError(format!("dequantize {dtype:?}: ys (len = {actual_output_len}) does not match the expected length of {expected_output_len}!")));
    }

    // Zip the blocks and outputs together
    Ok(xs.iter().zip(ys.chunks_exact_mut(block_size)).collect())
}

// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L224
pub fn make_qkx1_quants(nmax: i32, ntry: usize, x: &[f32]) -> (f32, f32) {
    let n = x.len();
    let mut l = vec![0; n];
    // Get min/max
    let min = *x
        .iter()
        .take(n)
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(&x[0]);
    let max = *x.iter().max_by(|a, b| a.total_cmp(b)).unwrap_or(&x[0]);

    // If min == max, all values are the same => nothing to do here
    if max == min {
        return (0.0, 0.0);
    }

    // Ensure min <= 0.0
    let mut min = min.min(0.);

    // Compute scale and inverse scale
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;

    for _ in 0..ntry {
        let mut sumlx = 0.0;
        let mut suml2 = 0;
        let mut did_change = false;

        for (i, value) in x.iter().enumerate().take(n) {
            let li = nearest_int(iscale * (value - min)).clamp(0, nmax);
            let clamped_li = li as u8;
            if clamped_li != l[i] {
                l[i] = clamped_li;
                did_change = true;
            }
            sumlx += (value - min) * li as f32;
            suml2 += li * li;
        }
        scale = sumlx / suml2 as f32;

        let sum: f32 = x
            .iter()
            .take(n)
            .zip(l.iter().take(n))
            .map(|(xi, &li)| xi - scale * li as f32)
            .sum();

        min = sum / n as f32;
        if min > 0.0 {
            min = 0.0;
        }
        iscale = 1.0 / scale;
        if !did_change {
            break;
        }
    }
    (scale, -min)
}

pub unsafe fn make_qx_quants(
    n: usize,
    nmax: i32,
    x: *const f32,
    ls: *mut i8,
    rmse_type: i32,
) -> f32 {
    let mut max = 0f32;
    let mut amax = 0f32;
    for i in 0..n {
        let x = *x.add(i);
        let ax = x.abs();
        if ax > amax {
            amax = ax;
            max = x;
        }
    }
    if amax == 0. {
        // all zero
        for i in 0..n {
            *ls.add(i) = 0;
        }
        return 0.;
    }
    let mut iscale = -(nmax as f32) / max;
    if rmse_type == 0 {
        for i in 0..n {
            let x = *x.add(i);
            let l = nearest_int(iscale * x);
            *ls.add(i) = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
        }
        return 1.0 / iscale;
    }
    let weight_type = rmse_type % 2;
    let mut sumlx = 0f32;
    let mut suml2 = 0f32;
    for i in 0..n {
        let x = *x.add(i);
        let l = nearest_int(iscale * x);
        let l = l.clamp(-nmax, nmax - 1);
        *ls.add(i) = (l + nmax) as i8;
        let w = if weight_type == 1 { x * x } else { 1.0 };
        let l = l as f32;
        sumlx += w * x * l;
        suml2 += w * l * l;
    }
    let mut scale = sumlx / suml2;
    let mut best = scale * sumlx;
    for _itry in 0..3 {
        let iscale = 1.0 / scale;
        let mut slx = 0f32;
        let mut sl2 = 0f32;
        let mut changed = false;
        for i in 0..n {
            let x = *x.add(i);
            let l = nearest_int(iscale * x);
            let l = l.clamp(-nmax, nmax - 1);
            if l + nmax != *ls.add(i) as i32 {
                changed = true;
            }
            let w = if weight_type == 1 { x * x } else { 1f32 };
            let l = l as f32;
            slx += w * x * l;
            sl2 += w * l * l;
        }
        if !changed || sl2 == 0.0 || slx * slx <= best * sl2 {
            break;
        }
        for i in 0..n {
            let x = *x.add(i);
            let l = nearest_int(iscale * x);
            *ls.add(i) = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
        }
        sumlx = slx;
        suml2 = sl2;
        scale = sumlx / suml2;
        best = scale * sumlx;
    }
    for _itry in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let x = *x.add(i);
            let w = if weight_type == 1 { x * x } else { 1. };
            let l = *ls.add(i) as i32 - nmax;
            let mut slx = sumlx - w * x * l as f32;
            if slx > 0. {
                let mut sl2 = suml2 - w * l as f32 * l as f32;
                let new_l = nearest_int(x * sl2 / slx);
                let new_l = new_l.clamp(-nmax, nmax - 1);
                if new_l != l {
                    slx += w * x * new_l as f32;
                    sl2 += w * new_l as f32 * new_l as f32;
                    if sl2 > 0. && slx * slx * suml2 > sumlx * sumlx * sl2 {
                        *ls.add(i) = (nmax + new_l) as i8;
                        sumlx = slx;
                        suml2 = sl2;
                        scale = sumlx / suml2;
                        best = scale * sumlx;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if rmse_type < 3 {
        return scale;
    }
    for is in -4..4 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1f32 * is as f32) / max;
        let mut sumlx = 0.;
        let mut suml2 = 0.;
        for i in 0..n {
            let x = *x.add(i);
            let l = nearest_int(iscale * x);
            let l = l.clamp(-nmax, nmax - 1);
            let w = if weight_type == 1 { x * x } else { 1. };
            let l = l as f32;
            sumlx += w * x * l;
            suml2 += w * l * l;
        }
        if suml2 > 0. && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let x = *x.add(i);
                let l = nearest_int(iscale * x);
                *ls.add(i) = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}
