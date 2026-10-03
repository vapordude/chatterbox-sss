use candle_core::{DType, Device, Result, Tensor};

/// Create mask for subsequent steps (size, size) with chunk size.
/// For streaming encoder.
///
/// Python code:
/// pos_idx = torch.arange(size, device=device)
/// block_value = (torch.div(pos_idx, chunk_size, rounding_mode='trunc') + 1) * chunk_size
/// ret = pos_idx.unsqueeze(0) < block_value.unsqueeze(1)
pub fn subsequent_chunk_mask(
    size: usize,
    chunk_size: usize,
    _num_left_chunks: isize,
    device: &Device,
) -> Result<Tensor> {
    // Generate 0..size
    let pos_idx = Tensor::arange(0u32, size as u32, device)?;

    // block_value = (pos_idx / chunk_size + 1) * chunk_size
    // Integer division is not natively supported directly on tensor with `/` like in Python,
    // so we can cast to float, divide, floor/trunc, then cast back, or do math manually.
    // However, since we're using u32, we can just do math using the elements or via scalar operators if available.
    // candle allows scalar div:
    let block_value_div = pos_idx.broadcast_div(&Tensor::new(chunk_size as u32, device)?)?;
    let block_value_add = block_value_div.broadcast_add(&Tensor::new(1u32, device)?)?;
    let block_value = block_value_add.broadcast_mul(&Tensor::new(chunk_size as u32, device)?)?;

    // pos_idx.unsqueeze(0)
    let pos_idx_u = pos_idx.unsqueeze(0)?;
    // block_value.unsqueeze(1)
    let block_value_u = block_value.unsqueeze(1)?;

    // Comparison pos_idx_u < block_value_u
    // We broadcast them to (size, size) first
    let pos_idx_b = pos_idx_u.broadcast_as((size, size))?;
    let block_value_b = block_value_u.broadcast_as((size, size))?;

    // Create a boolean mask tensor
    // Currently, candle's lt (less than) returns u8 (0 and 1).
    let ret = pos_idx_b.lt(&block_value_b)?;

    Ok(ret)
}

/// Apply optional mask for encoder
/// Python: add_optional_chunk_mask
pub fn add_optional_chunk_mask(
    xs: &Tensor,
    masks: &Tensor, // (B, 1, L)
    use_dynamic_chunk: bool,
    use_dynamic_left_chunk: bool,
    decoding_chunk_size: isize,
    static_chunk_size: isize,
    num_decoding_left_chunks: isize,
    enable_full_context: bool,
    device: &Device,
) -> Result<Tensor> {
    if use_dynamic_chunk {
        let max_len = xs.dims()[1];
        let mut chunk_size: usize;
        let mut num_left_chunks: isize = -1;

        if decoding_chunk_size < 0 {
            chunk_size = max_len;
        } else if decoding_chunk_size > 0 {
            chunk_size = decoding_chunk_size as usize;
            num_left_chunks = num_decoding_left_chunks;
        } else {
            // Training time: dynamic chunk size
            // For simplicity, we fallback to max_len (or we could use rand)
            chunk_size = std::cmp::max(1, max_len);
            if chunk_size > max_len / 2 && enable_full_context {
                chunk_size = max_len;
            } else {
                chunk_size = (chunk_size % 25) + 1;
                if use_dynamic_left_chunk {
                    let _max_left_chunks = (max_len.saturating_sub(1)) / chunk_size;
                    // simplified to 0 for mock rand
                    num_left_chunks = 0;
                }
            }
        }

        let chunk_masks = subsequent_chunk_mask(max_len, chunk_size, num_left_chunks, device)?;
        let chunk_masks_u = chunk_masks.unsqueeze(0)?; // (1, L, L)

        // chunk_masks = masks & chunk_masks (logical AND)
        // Masks are u8 containing 0s and 1s, chunk_masks is u8.
        // We can do element-wise minimum or multiply.
        // First broadcast `masks` (B, 1, L) and `chunk_masks_u` (1, L, L) to (B, L, L)
        let b = masks.dims()[0];
        let l = masks.dims()[2];
        let m_b = masks.broadcast_as((b, l, l))?;
        let c_b = chunk_masks_u.broadcast_as((b, l, l))?;

        // Element-wise minimum works as logical AND for u8 0/1 tensors
        let out_mask = m_b.minimum(&c_b)?;

        return Ok(out_mask);
    } else if static_chunk_size > 0 {
        let num_left_chunks = num_decoding_left_chunks;
        let max_len = xs.dims()[1];
        let chunk_masks =
            subsequent_chunk_mask(max_len, static_chunk_size as usize, num_left_chunks, device)?;
        let chunk_masks_u = chunk_masks.unsqueeze(0)?;

        let b = masks.dims()[0];
        let l = masks.dims()[2];
        let m_b = masks.broadcast_as((b, l, l))?;
        let c_b = chunk_masks_u.broadcast_as((b, l, l))?;

        let out_mask = m_b.minimum(&c_b)?;
        return Ok(out_mask);
    } else {
        return Ok(masks.clone());
    }
}

/// Make mask tensor containing indices of padded part.
/// Python: make_pad_mask
/// lengths: Batch of lengths (B,)
/// Returns mask: (B, max_len)
pub fn make_pad_mask(lengths: &Tensor, mut max_len: usize, device: &Device) -> Result<Tensor> {
    let lengths = lengths.to_dtype(DType::I64)?;
    let batch_size = lengths.dims()[0];

    if max_len == 0 {
        // max() on 1D tensor
        let max_t = lengths.max(0)?;
        // Extract scalar. max_t is 0D tensor now.
        let max_scalar = max_t.to_scalar::<i64>()?;
        max_len = max_scalar as usize;
    }

    let seq_range = Tensor::arange(0i64, max_len as i64, device)?;
    // seq_range_expand: (B, max_len)
    let seq_range_expand = seq_range
        .unsqueeze(0)?
        .broadcast_as((batch_size, max_len))?;

    // seq_length_expand: (B, 1) broadcasted to (B, max_len)
    let seq_length_expand = lengths.unsqueeze(1)?.broadcast_as((batch_size, max_len))?;

    // mask = seq_range_expand >= seq_length_expand
    let mask = seq_range_expand.ge(&seq_length_expand)?;

    Ok(mask)
}
