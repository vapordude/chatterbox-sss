use candle_core::{Result, Tensor};
use candle_nn::Linear;

/// Diagonal init as described in 3.3 https://arxiv.org/pdf/2510.07979
/// Translated from python `get_intmeanflow_time_mixer`.
pub fn get_intmeanflow_time_mixer(dims: usize, device: &candle_core::Device) -> Result<Linear> {
    // target_weight = torch.zeros(dims, 2 * dims)
    // target_weight[:, 0:dims] = torch.eye(dims)

    // Create an eye matrix of size dims x dims
    let eye = Tensor::eye(dims, candle_core::DType::F32, device)?;

    // Create zeros of size dims x dims for the right half
    let zeros = Tensor::zeros((dims, dims), candle_core::DType::F32, device)?;

    // Concatenate along the second dimension (columns)
    let target_weight = Tensor::cat(&[&eye, &zeros], 1)?;

    // Wrap it in Linear
    // candle_nn::Linear takes (weight, optional bias)
    Ok(Linear::new(target_weight, None))
}
