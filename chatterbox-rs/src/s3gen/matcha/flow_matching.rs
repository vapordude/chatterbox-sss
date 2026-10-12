use candle_core::{IndexOp, Result, Tensor};
use candle_nn::VarBuilder;

use super::decoder::Decoder;

/// BaseCFM is the base class for the Conditional Flow Matching implementation.
/// It contains the estimator (Decoder) and the core solving algorithm (Euler solver).
pub struct BaseCFM {
    pub n_feats: usize,
    pub n_spks: usize,
    pub spk_emb_dim: usize,
    pub sigma_min: f64,
    pub estimator: Decoder,
}

impl BaseCFM {
    pub fn new(
        n_feats: usize,
        n_spks: usize,
        spk_emb_dim: usize,
        sigma_min: Option<f64>,
        estimator: Decoder,
    ) -> Self {
        Self {
            n_feats,
            n_spks,
            spk_emb_dim,
            sigma_min: sigma_min.unwrap_or(1e-4),
            estimator,
        }
    }

    /// Forward diffusion wrapper. Creates random noise and linearly spaced timesteps
    /// and passes them to the ODE Euler solver.
    pub fn forward(
        &self,
        mu: &Tensor,
        mask: &Tensor,
        n_timesteps: usize,
        temperature: f64,
        spks: Option<&Tensor>,
        cond: Option<&Tensor>,
    ) -> Result<Tensor> {
        let z = (mu.randn_like(0.0, 1.0)? * temperature)?;

        let t_span_vec: Vec<f32> = (0..=n_timesteps)
            .map(|i| i as f32 / n_timesteps as f32)
            .collect();
        let t_span = Tensor::from_vec(t_span_vec, n_timesteps + 1, mu.device())?.to_dtype(mu.dtype())?;

        self.solve_euler(&z, &t_span, mu, mask, spks, cond)
    }

    /// Solves the Ordinary Differential Equation using a fixed Euler step method.
    pub fn solve_euler(
        &self,
        x: &Tensor,
        t_span: &Tensor,
        mu: &Tensor,
        mask: &Tensor,
        spks: Option<&Tensor>,
        cond: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut t = t_span.i(0)?;
        let mut dt = t_span.i(1)?.sub(&t_span.i(0)?)?;

        let mut current_x = x.clone();

        let len = t_span.dim(0)?;

        for step in 1..len {
            let b = current_x.dim(0)?;
            let t_b = t.broadcast_as((b,))?;

            let dphi_dt = self.estimator.forward(&current_x, mask, mu, &t_b, spks, cond)?;
            current_x = current_x.add(&(dt.broadcast_mul(&dphi_dt)?))?;
            t = t.add(&dt)?;

            if step < len - 1 {
                dt = t_span.i(step + 1)?.sub(&t)?;
            }
        }

        Ok(current_x)
    }
}

/// CFM is the Conditional Flow Matching struct, containing the initialized BaseCFM
/// wrapper and acting as the main endpoint.
pub struct CFM {
    pub base: BaseCFM,
}

impl CFM {
    pub fn new(
        in_channels: usize,
        out_channel: usize,
        n_spks: usize,
        spk_emb_dim: usize,
        sigma_min: Option<f64>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let estimator_in_channels = if n_spks > 1 {
            in_channels + spk_emb_dim
        } else {
            in_channels
        };

        let estimator = Decoder::new(estimator_in_channels, out_channel, vb.pp("estimator"))?;
        let base = BaseCFM::new(in_channels, n_spks, spk_emb_dim, sigma_min, estimator);

        Ok(Self { base })
    }

    /// Forward diffusion wrapper. Creates random noise and linearly spaced timesteps
    /// and passes them to the ODE Euler solver.
    pub fn forward(
        &self,
        mu: &Tensor,
        mask: &Tensor,
        n_timesteps: usize,
        temperature: f64,
        spks: Option<&Tensor>,
        cond: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.base.forward(mu, mask, n_timesteps, temperature, spks, cond)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_cfm_forward_and_solve_euler() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);

        let n_feats = 80;
        let n_spks = 1;
        let spk_emb_dim = 128;

        let cfm = CFM::new(n_feats, n_feats, n_spks, spk_emb_dim, Some(1e-4), vb.clone())?;

        let mu = Tensor::zeros((2, n_feats, 100), candle_core::DType::F32, &device)?;
        let mask = Tensor::ones((2, 1, 100), candle_core::DType::F32, &device)?;

        let out = cfm.forward(&mu, &mask, 10, 1.0, None, None)?;

        // Assert output shape matches mu
        assert_eq!(out.dims(), mu.dims());

        Ok(())
    }
}
