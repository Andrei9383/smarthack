//! K2 window gather. Python reference: `wattsentry.kernels.windows.window_gather_py` (numpy fancy-index gather).
//! Source is a C-contiguous (R, W) array; window b is the contiguous slice src[rows[b], starts[b]*step + offset ..]
//! [..length]. Bounds are re-checked here so a bad call cannot read out of range.

use rayon::prelude::*;
use std::sync::OnceLock;

/// numpy's npy_halfbits_to_floatbits for every float16 bit pattern (NaN payloads kept, no quiet bit added).
pub fn f16_table() -> &'static [u32] {
    static TABLE: OnceLock<Vec<u32>> = OnceLock::new();
    TABLE.get_or_init(|| (0..=u16::MAX).map(half_to_float_bits).collect())
}

fn half_to_float_bits(h: u16) -> u32 {
    let h = h as u32;
    let sign = (h & 0x8000) << 16;
    let exp = h & 0x7c00;
    let mant = h & 0x03ff;
    match exp {
        0x7c00 => sign | 0x7f80_0000 | (mant << 13), // inf / NaN
        0 if mant == 0 => sign,                      // signed zero
        0 => {
            // subnormal: normalise the mantissa
            let mut m = mant << 1;
            let mut e = 0u32;
            while m & 0x0400 == 0 {
                m <<= 1;
                e += 1;
            }
            sign | ((112 - e) << 23) | ((m & 0x03ff) << 13)
        }
        _ => sign | (((exp >> 10) + 112) << 23) | (mant << 13),
    }
}

/// Validate window b and return its first index in the flat source.
#[inline]
fn window_start(rows: &[i64], starts: &[i64], b: usize, step: i64, offset: i64, length: usize, r: usize,
                w: usize) -> Result<usize, String> {
    let row = rows[b];
    if row < 0 || row as usize >= r {
        return Err(format!("row {row} out of range [0, {r})"));
    }
    let lo = starts[b] * step + offset;
    if lo < 0 || lo as usize + length > w {
        return Err(format!("window [{lo}, {}) outside [0, {w})", lo + length as i64));
    }
    Ok(row as usize * w + lo as usize)
}

/// out[b, :] = conv(src[rows[b], starts[b]*step + offset ..][..length]); all windows are checked first.
#[allow(clippy::too_many_arguments)]
pub fn gather<S: Copy + Sync, O: Send>(src: &[S], r: usize, w: usize, rows: &[i64], starts: &[i64], step: i64,
                                       offset: i64, length: usize, out: &mut [O], conv: impl Fn(S) -> O + Sync)
    -> Result<(), String> {
    for b in 0..rows.len() {
        window_start(rows, starts, b, step, offset, length, r, w)?;
    }
    if length == 0 {
        return Ok(());
    }
    out.par_chunks_mut(length).enumerate().with_min_len(16).for_each(|(b, o)| {
        let a = window_start(rows, starts, b, step, offset, length, r, w).unwrap();
        for (d, &x) in o.iter_mut().zip(&src[a..a + length]) {
            *d = conv(x);
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_basics() {
        assert_eq!(f32::from_bits(half_to_float_bits(0x3c00)), 1.0);
        assert_eq!(f32::from_bits(half_to_float_bits(0xc000)), -2.0);
        assert_eq!(f32::from_bits(half_to_float_bits(0x0001)), 5.960_464_5e-8);
        assert_eq!(half_to_float_bits(0x7c01), 0x7f80_2000);
    }

    #[test]
    fn gather_small() {
        let src: Vec<u16> = (0..20).collect(); // 2 rows x 10
        let mut out = vec![0u16; 6];
        gather(&src, 2, 10, &[1, 0], &[2, 5], 1, -1, 3, &mut out, |x| x).unwrap();
        assert_eq!(out, vec![11, 12, 13, 4, 5, 6]);
        assert!(gather(&src, 2, 10, &[0], &[0], 1, -1, 3, &mut out[..3], |x| x).is_err());
    }
}
