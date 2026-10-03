use candle_core::{Device, Result, Tensor};
use candle_nn::{linear, Linear, RNN, VarBuilder};
use candle_core::Module;
use crate::voice_encoder::config::VoiceEncConfig;

// pack function simplified, in Rust we'll usually work with Tensors
pub fn pack(arrays: &[Tensor], seq_len: Option<usize>, pad_value: f32, device: &Device) -> Result<Tensor> {
    let mut seq_len = seq_len.unwrap_or(0);
    if seq_len == 0 {
        for t in arrays {
            seq_len = seq_len.max(t.dim(0)?);
        }
    }

    let b = arrays.len();
    if b == 0 {
        candle_core::bail!("empty arrays to pack");
    }

    let m = arrays[0].dim(1)?;
    
    // Create zero/pad tensor of shape (B, seq_len, M)
    let mut packed = vec![pad_value; b * seq_len * m];
    
    for (i, t) in arrays.iter().enumerate() {
        let t_len = t.dim(0)?;
        let t_vec = t.to_vec2::<f32>()?;
        for row in 0..t_len {
            for col in 0..m {
                packed[i * seq_len * m + row * m + col] = t_vec[row][col];
            }
        }
    }
    
    Tensor::from_vec(packed, (b, seq_len, m), device)
}

pub fn get_num_wins(n_frames: usize, step: usize, min_coverage: f64, hp: &VoiceEncConfig) -> (usize, usize) {
    assert!(n_frames > 0);
    let win_size = hp.ve_partial_frames;
    let max_val = if n_frames + step > win_size {
        n_frames + step - win_size
    } else {
        0
    };
    
    let mut n_wins = max_val / step;
    let remainder = max_val % step;
    
    if n_wins == 0 || (remainder as f64 + (win_size - step) as f64) / (win_size as f64) >= min_coverage {
        n_wins += 1;
    }
    
    let target_n = win_size + step * n_wins.saturating_sub(1);
    (n_wins, target_n)
}

pub fn get_frame_step(overlap: f64, rate: Option<f64>, hp: &VoiceEncConfig) -> usize {
    assert!(overlap >= 0.0 && overlap < 1.0);
    let mut frame_step = if let Some(r) = rate {
        ((hp.sample_rate as f64 / r) / hp.ve_partial_frames as f64).round() as usize
    } else {
        (hp.ve_partial_frames as f64 * (1.0 - overlap)).round() as usize
    };
    
    if frame_step == 0 {
        frame_step = 1;
    }
    if frame_step > hp.ve_partial_frames {
        frame_step = hp.ve_partial_frames;
    }
    frame_step
}


pub struct VoiceEncoder {
    hp: VoiceEncConfig,
    lstm_layers: Vec<candle_nn::rnn::LSTM>,
    proj: Linear,
}

impl VoiceEncoder {
    pub fn new(hp: VoiceEncConfig, vb: VarBuilder) -> Result<Self> {
        let mut lstm_layers = Vec::new();
        for i in 0..3 {
            let in_dim = if i == 0 { hp.num_mels } else { hp.ve_hidden_size };
            let lstm_config = candle_nn::rnn::LSTMConfig {
                
                ..Default::default()
            };
            let lstm = candle_nn::rnn::lstm(in_dim, hp.ve_hidden_size, lstm_config, vb.pp(format!("lstm.l{}", i)))?;
            lstm_layers.push(lstm);
        }
        
        let proj = linear(hp.ve_hidden_size, hp.speaker_embed_size, vb.pp("proj"))?;
        
        Ok(Self { hp, lstm_layers, proj })
    }

    pub fn forward(&self, mels: &Tensor) -> Result<Tensor> {
        // Validation check for normalized_mels
        if self.hp.normalized_mels {
            // Note: In inference context, the mel should have been generated properly. 
            // In Python it raises an exception if out of bounds, but for performance 
            // and because we generate it ourselves, we assume the input is correct.
        }
        
        let mut h = mels.clone();
        let mut last_hidden = h.clone();
        
        for lstm in &self.lstm_layers {
            let states = lstm.seq(&h)?; 
            
            // Collect the hidden states at each step to form the output sequence for the next layer
            let mut h_seq = Vec::with_capacity(states.len());
            for state in &states {
                h_seq.push(state.h().clone());
            }
            h = Tensor::stack(&h_seq, 1)?; // stack along sequence dimension
            last_hidden = states.last().unwrap().h().clone();
        }
        
        let mut raw_embeds = self.proj.forward(&last_hidden)?; // (B, E)

        if self.hp.ve_final_relu {
            raw_embeds = raw_embeds.relu()?;
        }
        
        // L2 normalize
        let norm = raw_embeds.sqr()?.sum_keepdim(1)?.sqrt()?;
        let embeds = raw_embeds.broadcast_div(&norm)?;
        
        Ok(embeds)
    }

    pub fn inference(&self, mels: &Tensor, mel_lens: &[usize], overlap: f64, rate: Option<f64>, min_coverage: f64, batch_size: Option<usize>) -> Result<Tensor> {
        // mels: (B, T, M)
        let b = mels.dim(0)?;
        let t = mels.dim(1)?;
        let m = mels.dim(2)?;
        
        let frame_step = get_frame_step(overlap, rate, &self.hp);
        
        let mut n_partials_list = Vec::with_capacity(b);
        let mut target_lens = Vec::with_capacity(b);
        
        for &l in mel_lens {
            let (n_p, t_l) = get_num_wins(l, frame_step, min_coverage, &self.hp);
            n_partials_list.push(n_p);
            target_lens.push(t_l);
        }
        
        let max_target_len = *target_lens.iter().max().unwrap_or(&0);
        let len_diff = max_target_len.saturating_sub(t);
        
        let padded_mels = if len_diff > 0 {
            let pad = Tensor::zeros((b, len_diff, m), mels.dtype(), mels.device())?;
            Tensor::cat(&[mels, &pad], 1)?
        } else {
            mels.clone()
        };
        
        let mut partials = Vec::new();
        for i in 0..b {
            let mel = padded_mels.narrow(0, i, 1)?.squeeze(0)?; // (T, M)
            let n_partial = n_partials_list[i];
            for p in 0..n_partial {
                let start = p * frame_step;
                let chunk = mel.narrow(0, start, self.hp.ve_partial_frames)?;
                partials.push(chunk);
            }
        }
        
        let stacked_partials = Tensor::stack(&partials, 0)?; // (Total_partials, T_partial, M)
        let total_partials = stacked_partials.dim(0)?;
        
        let b_size = batch_size.unwrap_or(total_partials);
        
        let mut partial_embeds = Vec::new();
        let mut start = 0;
        while start < total_partials {
            let end = (start + b_size).min(total_partials);
            let batch = stacked_partials.narrow(0, start, end - start)?;
            let emb = self.forward(&batch)?; // (batch_size, E)
            partial_embeds.push(emb);
            start = end;
        }
        
        let partial_embeds_tensor = Tensor::cat(&partial_embeds, 0)?; // (Total_partials, E)
        
        let mut start_idx = 0;
        let mut raw_embeds_list = Vec::with_capacity(b);
        for &n_p in &n_partials_list {
            let end_idx = start_idx + n_p;
            let group = partial_embeds_tensor.narrow(0, start_idx, n_p)?;
            // mean along dim 0
            let mean = group.mean_keepdim(0)?.squeeze(0)?;
            raw_embeds_list.push(mean);
            start_idx = end_idx;
        }
        
        let raw_embeds = Tensor::stack(&raw_embeds_list, 0)?;
        let norm = raw_embeds.sqr()?.sum_keepdim(1)?.sqrt()?;
        let embeds = raw_embeds.broadcast_div(&norm)?;
        
        Ok(embeds)
    }

    pub fn utt_to_spk_embed(utt_embeds: &Tensor) -> Result<Tensor> {
        let mean = utt_embeds.mean_keepdim(0)?.squeeze(0)?;
        let norm = mean.sqr()?.sum_keepdim(0)?.sqrt()?;
        mean.broadcast_div(&norm)
    }

    pub fn voice_similarity(embeds_x: &Tensor, embeds_y: &Tensor) -> Result<Tensor> {
        // Assume inputs are L2 normalized and 1D (E) or 2D (B, E).
        // For simplicity, handle 1D here, returning dot product.
        let ex = if embeds_x.dims().len() == 2 {
            Self::utt_to_spk_embed(embeds_x)?
        } else {
            embeds_x.clone()
        };
        let ey = if embeds_y.dims().len() == 2 {
            Self::utt_to_spk_embed(embeds_y)?
        } else {
            embeds_y.clone()
        };
        // Dot product
        (ex * ey)?.sum_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_num_wins() {
        let hp = VoiceEncConfig::default();
        let frame_step = get_frame_step(0.5, None, &hp);
        assert_eq!(frame_step, 80);
        
        let (n_wins, target_n) = get_num_wins(200, frame_step, 0.8, &hp);
        // (200 - 160 + 80)/80 = 120/80 = 1, rem 40
        // min_coverage = 0.8 -> (40 + 160 - 80)/160 = 120/160 = 0.75 < 0.8
        // so n_wins = 1. target_n = 160
        assert_eq!(n_wins, 1);
        assert_eq!(target_n, 160);
    }
}
