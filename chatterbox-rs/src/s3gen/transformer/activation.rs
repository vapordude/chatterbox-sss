use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

#[derive(Debug)]
pub struct Swish;

impl Swish {
    pub fn new() -> Self {
        Self
    }
}

impl Module for Swish {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let sigmoid_x = candle_nn::ops::sigmoid(x)?;
        x.broadcast_mul(&sigmoid_x)
    }
}

#[derive(Debug)]
pub struct Snake {
    alpha: Tensor,
    alpha_logscale: bool,
    no_div_by_zero: f64,
}

impl Snake {
    pub fn new(
        in_features: usize,
        alpha: f64,
        alpha_logscale: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let alpha_tensor = if alpha_logscale {
            vb.get_with_hints(
                in_features,
                "alpha",
                candle_nn::Init::Const(0.0), // log scale alphas initialized to zeros
            )?
        } else {
            vb.get_with_hints(
                in_features,
                "alpha",
                candle_nn::Init::Const(alpha), // linear scale alphas initialized to alpha
            )?
        };

        Ok(Self {
            alpha: alpha_tensor,
            alpha_logscale,
            no_div_by_zero: 1e-9,
        })
    }
}

impl Module for Snake {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x is expected to have shape [B, C, T]
        // alpha has shape [C]
        // We need to reshape alpha to [1, C, 1] to broadcast with x
        let alpha = self.alpha.unsqueeze(0)?.unsqueeze(2)?;

        let alpha = if self.alpha_logscale {
            alpha.exp()?
        } else {
            alpha
        };

        let x_alpha = x.broadcast_mul(&alpha)?;
        let sin_x_alpha = x_alpha.sin()?;
        let sin_sq = sin_x_alpha.sqr()?;

        let denom = (alpha + self.no_div_by_zero)?;
        let term2 = sin_sq.broadcast_div(&denom)?;

        x.broadcast_add(&term2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::VarMap;

    #[test]
    fn test_swish() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &device)?;
        let swish = Swish::new();
        let y = swish.forward(&x)?;

        let expected_y = [
            -2.0 * (-2.0f32).exp() / (1.0 + (-2.0f32).exp()),
            -1.0 * (-1.0f32).exp() / (1.0 + (-1.0f32).exp()),
            0.0,
            1.0 / (1.0 + (-1.0f32).exp()),
            2.0 / (1.0 + (-2.0f32).exp()),
        ];

        let y_vec = y.to_vec1::<f32>()?;
        for (a, b) in y_vec.iter().zip(expected_y.iter()) {
            assert!((a - b).abs() < 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_snake_linear() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let in_features = 2;
        let snake = Snake::new(in_features, 1.0, false, vb)?;

        // Input shape [B, C, T] = [1, 2, 3]
        let x = Tensor::new(&[[[0.1f32, 0.2, 0.3], [0.4, 0.5, 0.6]]], &device)?;

        let y = snake.forward(&x)?;

        let y_vec = y.flatten_all()?.to_vec1::<f32>()?;
        let x_vec = x.flatten_all()?.to_vec1::<f32>()?;

        for (i, val) in x_vec.iter().enumerate() {
            let expected = val + (val.sin().powi(2) / (1.0 + 1e-9));
            assert!((y_vec[i] - expected).abs() < 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_snake_logscale() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let in_features = 2;
        let snake = Snake::new(in_features, 1.0, true, vb)?;

        // Input shape [B, C, T] = [1, 2, 3]
        let x = Tensor::new(&[[[0.1f32, 0.2, 0.3], [0.4, 0.5, 0.6]]], &device)?;

        let y = snake.forward(&x)?;

        let y_vec = y.flatten_all()?.to_vec1::<f32>()?;
        let x_vec = x.flatten_all()?.to_vec1::<f32>()?;

        for (i, val) in x_vec.iter().enumerate() {
            // alpha is initialized to 0, so exp(alpha) = 1.0
            let expected = val + (val.sin().powi(2) / (1.0 + 1e-9));
            assert!((y_vec[i] - expected).abs() < 1e-5);
        }

        Ok(())
    }
}
