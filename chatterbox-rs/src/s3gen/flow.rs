use candle_core::{Module, Result, Tensor, DType};
use candle_nn::{Embedding, Linear};
use super::utils::mask::make_pad_mask;

pub trait Encoder {
    fn forward(&self, token: &Tensor, token_len: &Tensor) -> Result<(Tensor, Tensor)>;
    fn output_size(&self) -> usize;
}

pub trait FlowDecoder {
    fn forward(
        &self,
        mu: &Tensor,
        mask: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
        n_timesteps: usize,
        noised_mels: Option<&Tensor>,
        meanflow: bool,
    ) -> Result<(Tensor, Option<Tensor>)>;
}

pub struct CausalMaskedDiffWithXvec<E: Encoder, D: FlowDecoder> {
    pub input_size: usize,
    pub output_size: usize,
    pub vocab_size: usize,
    pub output_type: String,
    pub input_frame_rate: usize,
    pub only_mask_loss: bool,
    pub token_mel_ratio: usize,
    pub pre_lookahead_len: usize,

    pub input_embedding: Embedding,
    pub spk_embed_affine_layer: Linear,
    pub encoder: E,
    pub encoder_proj: Linear,
    pub decoder: D,
}

impl<E: Encoder, D: FlowDecoder> CausalMaskedDiffWithXvec<E, D> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_size: usize,
        output_size: usize,
        _spk_embed_dim: usize,
        vocab_size: usize,
        output_type: String,
        input_frame_rate: usize,
        only_mask_loss: bool,
        token_mel_ratio: usize,
        pre_lookahead_len: usize,
        input_embedding: Embedding,
        spk_embed_affine_layer: Linear,
        encoder: E,
        encoder_proj: Linear,
        decoder: D,
    ) -> Self {
        Self {
            input_size,
            output_size,
            vocab_size,
            output_type,
            input_frame_rate,
            only_mask_loss,
            token_mel_ratio,
            pre_lookahead_len,
            input_embedding,
            spk_embed_affine_layer,
            encoder,
            encoder_proj,
            decoder,
        }
    }

    fn repeat_batch_dim(tnsr: &Tensor, b: usize, ndim: usize) -> Result<Tensor> {
        let mut curr = tnsr.clone();
        while curr.dims().len() < ndim {
            curr = curr.unsqueeze(0)?;
        }
        if b > 1 && curr.dims()[0] == 1 {
            // Repeat batch dim
            let mut dims = curr.dims().to_vec();
            dims[0] = b;
            curr = curr.broadcast_as(dims.as_slice())?;
        }
        Ok(curr)
    }

    fn normalize_l2(x: &Tensor, dim: usize) -> Result<Tensor> {
        let sum_sq = x.sqr()?.sum_keepdim(dim)?;
        let norm = sum_sq.sqrt()?;
        let eps = Tensor::new(1e-12f32, x.device())?.to_dtype(x.dtype())?;
        let norm = norm.maximum(&eps)?;
        x.broadcast_div(&norm)
    }

    pub fn inference(
        &self,
        token: &Tensor,
        token_len: &Tensor,
        prompt_token: &Tensor,
        prompt_token_len: &Tensor,
        prompt_feat: &Tensor,
        _prompt_feat_len: &Tensor,
        embedding: &Tensor,
        finalize: bool,
        n_timesteps: usize,
        noised_mels: Option<&Tensor>,
        meanflow: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let b = token.dims()[0];
        let device = token.device();

        let mut emb = if embedding.dims().len() == 1 {
            embedding.unsqueeze(0)?
        } else {
            embedding.clone()
        };
        emb = Self::normalize_l2(&emb, 1)?;
        emb = self.spk_embed_affine_layer.forward(&emb)?;

        let pt = Self::repeat_batch_dim(prompt_token, b, 2)?;
        let pt_len = Self::repeat_batch_dim(prompt_token_len, b, 1)?;
        let pf = Self::repeat_batch_dim(prompt_feat, b, 3)?;
        let e = Self::repeat_batch_dim(&emb, b, 2)?;

        let t_cat = Tensor::cat(&[&pt, token], 1)?;
        let t_len_cat = pt_len.add(token_len)?;

        let max_len = t_cat.dims()[1];
        let pm = make_pad_mask(&t_len_cat, max_len, device)?; 
        let pm_f32 = Tensor::ones_like(&pm)?.to_dtype(DType::F32)?.sub(&pm.to_dtype(DType::F32)?)?;
        let mask = pm_f32.unsqueeze(2)?; 

        let mut t_embed = self.input_embedding.forward(&t_cat.to_dtype(DType::U32)?)?; 
        t_embed = t_embed.broadcast_mul(&mask.to_dtype(t_embed.dtype())?)?;

        let (mut h, h_masks) = self.encoder.forward(&t_embed, &t_len_cat)?;
        if !finalize {
            let drop_len = self.pre_lookahead_len * self.token_mel_ratio;
            let current_len = h.dims()[1];
            if current_len > drop_len {
                h = h.narrow(1, 0, current_len - drop_len)?;
            }
        }

        let h_lengths = h_masks.sum(2)?.squeeze(2)?;
        let mel_len1 = pf.dims()[1];
        let mel_len2 = h.dims()[1] - mel_len1;
        
        let h_proj = self.encoder_proj.forward(&h)?;

        let mut _conds_zeros = Tensor::zeros((b, mel_len1 + mel_len2, self.output_size), h_proj.dtype(), device)?;
        let conds = if mel_len2 > 0 {
            let z = Tensor::zeros((b, mel_len2, self.output_size), h_proj.dtype(), device)?;
            Tensor::cat(&[&pf, &z], 1)?
        } else {
            pf.clone()
        };
        let conds_t = conds.transpose(1, 2)?; 

        let h_max = h_proj.dims()[1];
        let h_pm = make_pad_mask(&h_lengths, h_max, device)?;
        let h_pm_f32 = Tensor::ones_like(&h_pm)?.to_dtype(DType::F32)?.sub(&h_pm.to_dtype(DType::F32)?)?;
        let h_mask = h_pm_f32.unsqueeze(1)?; 

        let h_proj_t = h_proj.transpose(1, 2)?; 

        let (feat, _) = self.decoder.forward(
            &h_proj_t.contiguous()?,
            &h_mask,
            &e,
            &conds_t,
            n_timesteps,
            noised_mels,
            meanflow,
        )?;

        let feat_out = feat.narrow(2, mel_len1, mel_len2)?;
        Ok((feat_out, None))
    }
}
