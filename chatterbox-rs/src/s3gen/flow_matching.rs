use candle_core::{Result, Tensor};

pub struct CfmParams {
    pub sigma_min: f64,
    pub solver: String,
    pub t_scheduler: String,
    pub training_cfg_rate: f64,
    pub inference_cfg_rate: f64,
    pub reg_loss_type: String,
}

impl Default for CfmParams {
    fn default() -> Self {
        Self {
            sigma_min: 1e-06,
            solver: "euler".to_string(),
            t_scheduler: "cosine".to_string(),
            training_cfg_rate: 0.2,
            inference_cfg_rate: 0.7,
            reg_loss_type: "l1".to_string(),
        }
    }
}

pub trait Estimator {
    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
        r: Option<&Tensor>,
    ) -> Result<Tensor>;
}

pub struct ConditionalCfm<E: Estimator> {
    pub n_feats: usize,
    pub cfm_params: CfmParams,
    pub n_spks: usize,
    pub spk_emb_dim: usize,
    pub estimator: E,
}

impl<E: Estimator> ConditionalCfm<E> {
    pub fn new(
        in_channels: usize,
        cfm_params: CfmParams,
        n_spks: usize,
        spk_emb_dim: usize,
        estimator: E,
    ) -> Self {
        Self {
            n_feats: in_channels,
            cfm_params,
            n_spks,
            spk_emb_dim,
            estimator,
        }
    }

    pub fn solve_euler(
        &self,
        x: &Tensor,
        t_span: &Tensor,
        mu: &Tensor,
        mask: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
        meanflow: bool,
    ) -> Result<Tensor> {
        let b = mu.dims()[0];
        let device = x.device();

        let mut x_curr = x.clone();

        let t_span_vec = t_span.to_vec1::<f32>()?;

        for i in 0..(t_span_vec.len() - 1) {
            let t_val = t_span_vec[i];
            let r_val = t_span_vec[i + 1];

            let t_scalar = Tensor::new(t_val, device)?;
            let r_scalar = Tensor::new(r_val, device)?;

            let t_tensor = t_scalar.broadcast_as((b,))?;
            let r_tensor = r_scalar.broadcast_as((b,))?;

            let x_in = Tensor::cat(&[&x_curr, &x_curr], 0)?;
            let mask_in = Tensor::cat(&[mask, mask], 0)?;
            
            let mu_zeros = Tensor::zeros_like(mu)?;
            let mu_in = Tensor::cat(&[mu, &mu_zeros], 0)?;
            
            let t_in = Tensor::cat(&[&t_tensor, &t_tensor], 0)?;
            let spks_in = Tensor::cat(&[spks, spks], 0)?;
            let cond_in = Tensor::cat(&[cond, cond], 0)?;

            let r_in = if meanflow {
                Some(Tensor::cat(&[&r_tensor, &r_tensor], 0)?)
            } else {
                None
            };

            let dxdt = self.estimator.forward(
                &x_in,
                &mask_in,
                &mu_in,
                &t_in,
                &spks_in,
                &cond_in,
                r_in.as_ref(),
            )?;

            let chunks = dxdt.chunk(2, 0)?;
            let dxdt_main = &chunks[0];
            let cfg_dxdt = &chunks[1];

            let inf_cfg = self.cfm_params.inference_cfg_rate;
            
            let term1 = dxdt_main.affine(1.0 + inf_cfg, 0.0)?;
            let term2 = cfg_dxdt.affine(inf_cfg, 0.0)?;
            
            let dxdt_adj = term1.sub(&term2)?;
            
            let dt = (r_val - t_val) as f64;
            x_curr = x_curr.add(&dxdt_adj.affine(dt, 0.0)?)?;
        }

        Ok(x_curr)
    }
}

pub struct CausalConditionalCfm<E: Estimator> {
    pub conditional_cfm: ConditionalCfm<E>,
}

impl<E: Estimator> CausalConditionalCfm<E> {
    pub fn new(
        in_channels: usize,
        cfm_params: CfmParams,
        n_spks: usize,
        spk_emb_dim: usize,
        estimator: E,
    ) -> Self {
        Self {
            conditional_cfm: ConditionalCfm::new(in_channels, cfm_params, n_spks, spk_emb_dim, estimator),
        }
    }

    pub fn forward(
        &self,
        mu: &Tensor,
        mask: &Tensor,
        n_timesteps: usize,
        _temperature: f64,
        spks: &Tensor,
        cond: &Tensor,
        noised_mels: Option<&Tensor>,
        meanflow: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let device = mu.device();
        let dtype = mu.dtype();
        
        let mut z = Tensor::randn(0.0f32, 1.0f32, mu.shape(), device)?.to_dtype(dtype)?;

        if let Some(noised_mels) = noised_mels {
            let prompt_len = mu.dims()[2] - noised_mels.dims()[2];
            let z_left = z.narrow(2, 0, prompt_len)?;
            z = Tensor::cat(&[&z_left, noised_mels], 2)?;
        }

        let step = 1.0 / (n_timesteps as f32);
        let mut t_span_vals = Vec::with_capacity(n_timesteps + 1);
        for i in 0..=n_timesteps {
            t_span_vals.push(i as f32 * step);
        }
        let mut t_span = Tensor::from_vec(t_span_vals.clone(), (n_timesteps + 1,), device)?;

        if !meanflow && self.conditional_cfm.cfm_params.t_scheduler == "cosine" {
            let half_pi = std::f32::consts::PI * 0.5;
            let t_span_f32 = t_span.to_vec1::<f32>()?;
            let mut cos_vals = Vec::with_capacity(n_timesteps + 1);
            for v in t_span_f32 {
                cos_vals.push(1.0 - (v * half_pi).cos());
            }
            t_span = Tensor::from_vec(cos_vals, (n_timesteps + 1,), device)?;
        }

        if meanflow {
            let out = self.basic_euler(&z, &t_span, mu, mask, spks, cond)?;
            Ok((out, None))
        } else {
            let out = self.conditional_cfm.solve_euler(&z, &t_span, mu, mask, spks, cond, meanflow)?;
            Ok((out, None))
        }
    }

    pub fn basic_euler(
        &self,
        x: &Tensor,
        t_span: &Tensor,
        mu: &Tensor,
        mask: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
    ) -> Result<Tensor> {
        let mut x_curr = x.clone();
        let t_span_vec = t_span.to_vec1::<f32>()?;
        let b = mu.dims()[0];
        let device = x.device();

        for i in 0..(t_span_vec.len() - 1) {
            let t_val = t_span_vec[i];
            let r_val = t_span_vec[i + 1];

            let t_scalar = Tensor::new(t_val, device)?;
            let r_scalar = Tensor::new(r_val, device)?;

            let t_tensor = t_scalar.broadcast_as((b,))?;
            let r_tensor = r_scalar.broadcast_as((b,))?;

            let dxdt = self.conditional_cfm.estimator.forward(
                &x_curr,
                mask,
                mu,
                &t_tensor,
                spks,
                cond,
                Some(&r_tensor),
            )?;

            let dt = (r_val - t_val) as f64;
            x_curr = x_curr.add(&dxdt.affine(dt, 0.0)?)?;
        }

        Ok(x_curr)
    }
}
