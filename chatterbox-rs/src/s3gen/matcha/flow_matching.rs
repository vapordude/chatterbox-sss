use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::s3gen::flow_matching::CfmParams;
use crate::s3gen::matcha::decoder::Decoder;

pub struct BaseCfm {
    pub n_feats: usize,
    pub n_spks: usize,
    pub spk_emb_dim: usize,
    pub solver: String,
    pub sigma_min: f64,
}

impl BaseCfm {
    pub fn new(n_feats: usize, cfm_params: &CfmParams, n_spks: usize, spk_emb_dim: usize) -> Self {
        Self {
            n_feats,
            n_spks,
            spk_emb_dim,
            solver: cfm_params.solver.clone(),
            sigma_min: cfm_params.sigma_min,
        }
    }
}

pub struct Cfm {
    pub base: BaseCfm,
    pub estimator: Decoder,
}

impl Cfm {
    pub fn new(
        in_channels: usize,
        out_channel: usize,
        cfm_params: &CfmParams,
        // decoder_params: &DecoderParams, // TODO actual decoder params if needed
        n_spks: usize,
        spk_emb_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let base = BaseCfm::new(in_channels, cfm_params, n_spks, spk_emb_dim);

        let in_channels_dec = in_channels + if n_spks > 1 { spk_emb_dim } else { 0 };
        let estimator = Decoder::new(in_channels_dec, out_channel, vb.pp("estimator"))?;

        Ok(Self { base, estimator })
    }

    pub fn forward(
        &self,
        mu: &Tensor,
        mask: &Tensor,
        n_timesteps: usize,
        temperature: f64,
        spks: Option<&Tensor>,
        cond: Option<&Tensor>,
    ) -> Result<Tensor> {
        let device = mu.device();
        let dtype = mu.dtype();

        let z = (Tensor::randn(0.0f32, 1.0f32, mu.shape(), device)? * temperature)?.to_dtype(dtype)?;

        let mut t_span_vec = Vec::with_capacity(n_timesteps + 1);
        for i in 0..=n_timesteps {
            t_span_vec.push(i as f64 / n_timesteps as f64);
        }
        let t_span = Tensor::new(t_span_vec.as_slice(), device)?.to_dtype(dtype)?;

        self.solve_euler(&z, &t_span, mu, mask, spks, cond)
    }

    pub fn solve_euler(
        &self,
        x: &Tensor,
        t_span: &Tensor,
        mu: &Tensor,
        mask: &Tensor,
        spks: Option<&Tensor>,
        cond: Option<&Tensor>,
    ) -> Result<Tensor> {
        let t_span_vals = t_span.to_vec1::<f32>()?;
        let mut t = t_span_vals[0];
        let mut dt = t_span_vals[t_span_vals.len() - 1] - t_span_vals[0];
        if t_span_vals.len() > 1 {
            dt = t_span_vals[1] - t_span_vals[0];
        }

        let mut x_curr = x.clone();
        let device = x.device();
        let dtype = x.dtype();

        for step in 1..t_span_vals.len() {
            // estimator call
            let t_tensor = Tensor::new(&[t], device)?.to_dtype(dtype)?;
            let dphi_dt = self.estimator.forward(&x_curr, mask, mu, &t_tensor, spks, cond)?;

            x_curr = (&x_curr + (&dphi_dt * dt as f64)?)?;
            t = t + dt;

            if step < t_span_vals.len() - 1 {
                dt = t_span_vals[step + 1] - t;
            }
        }

        Ok(x_curr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    #[test]
    fn test_cfm_initialization() -> Result<()> {
        let vb = VarBuilder::zeros(candle_core::DType::F32, &Device::Cpu);
        let cfm_params = CfmParams::default();
        let cfm = Cfm::new(80, 80, &cfm_params, 1, 64, vb)?;

        assert_eq!(cfm.base.n_feats, 80);
        assert_eq!(cfm.base.n_spks, 1);
        assert_eq!(cfm.base.spk_emb_dim, 64);
        assert_eq!(cfm.base.solver, "euler");

        Ok(())
    }

    #[test]
    fn test_cfm_forward() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let cfm_params = CfmParams::default();
        let cfm = Cfm::new(80, 80, &cfm_params, 1, 64, vb)?;

        let mu = Tensor::zeros((1, 80, 100), candle_core::DType::F32, &device)?;
        let mask = Tensor::ones((1, 1, 100), candle_core::DType::F32, &device)?;

        let output = cfm.forward(&mu, &mask, 10, 1.0, None, None)?;

        assert_eq!(output.shape().dims(), &[1, 80, 100]);
        Ok(())
    }
}
