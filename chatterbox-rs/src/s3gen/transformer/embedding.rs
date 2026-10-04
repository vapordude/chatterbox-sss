use candle_core::{DType, Result, Tensor};


pub struct PositionalEncoding {
    pub d_model: usize,
    xscale: f32,
    max_len: usize,
    pe: Tensor,
    pub dropout_rate: f32, // Simplified without full Dropout module for inference
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_len: usize, reverse: bool, device: &candle_core::Device) -> Result<Self> {
        let mut pe_data = vec![0.0f32; max_len * d_model];
        
        for pos in 0..max_len {
            for i in (0..d_model).step_by(2) {
                let div_term = (-((i as f32) * (10000.0f32.ln() / (d_model as f32)))).exp();
                let val = (pos as f32) * div_term;
                
                let sin_val = val.sin();
                let cos_val = val.cos();
                
                if reverse {
                    // pe[:, 0::2] = sin, pe[:, 1::2] = cos (as implemented in standard PositionalEncoding)
                    // Wait, standard PE does not have `reverse` flag like this. Let's look at Python.
                    // Oh, reverse is used in RelPositionalEncoding. Wait, the python implementation says:
                    // self.pe[:, 0::2] = torch.sin(...)
                    // self.pe[:, 1::2] = torch.cos(...)
                    // `reverse` simply changes nothing in PositionalEncoding's initial PE creation!
                }
                
                pe_data[pos * d_model + i] = sin_val;
                if i + 1 < d_model {
                    pe_data[pos * d_model + i + 1] = cos_val;
                }
            }
        }
        
        let pe = Tensor::from_vec(pe_data, (1, max_len, d_model), device)?;
        
        Ok(Self {
            d_model,
            xscale: (d_model as f32).sqrt(),
            max_len,
            pe,
            dropout_rate: 0.0,
        })
    }
    
    pub fn forward(&self, x: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let size = x.dim(1)?;
        let pos_emb = self.position_encoding(offset, size)?;
        
        // x = x * self.xscale + pos_emb
        let x_scaled = x.affine(self.xscale as f64, 0.0)?;
        let x_out = x_scaled.broadcast_add(&pos_emb)?;
        
        Ok((x_out, pos_emb))
    }
    
    pub fn position_encoding(&self, offset: usize, size: usize) -> Result<Tensor> {
        if offset + size > self.max_len {
            candle_core::bail!("offset + size exceeds max_len");
        }
        self.pe.narrow(1, offset, size)
    }
}

pub struct RelPositionalEncoding {
    pe: PositionalEncoding,
}

impl RelPositionalEncoding {
    pub fn new(d_model: usize, max_len: usize, device: &candle_core::Device) -> Result<Self> {
        Ok(Self {
            pe: PositionalEncoding::new(d_model, max_len, true, device)?,
        })
    }
    
    pub fn forward(&self, x: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let x_scaled = x.affine(self.pe.xscale as f64, 0.0)?;
        let pos_emb = self.pe.position_encoding(offset, x.dim(1)?)?;
        Ok((x_scaled, pos_emb))
    }
}

// WhisperPositionalEncoding, LearnablePositionalEncoding, NoPositionalEncoding, EspnetRelPositionalEncoding omitted for now unless needed, or can be added if required.

pub struct NoPositionalEncoding {
    d_model: usize,
}

impl NoPositionalEncoding {
    pub fn new(d_model: usize) -> Result<Self> {
        Ok(Self { d_model })
    }

    pub fn forward(&self, x: &Tensor, _offset: usize) -> Result<(Tensor, Tensor)> {
        let size = x.dim(1)?;
        let pos_emb = self.position_encoding(_offset, size, x.device())?;
        Ok((x.clone(), pos_emb))
    }

    pub fn position_encoding(&self, _offset: usize, size: usize, device: &candle_core::Device) -> Result<Tensor> {
        Tensor::zeros((1, size, self.d_model), DType::F32, device)
    }
}

pub struct EspnetRelPositionalEncoding {
    d_model: usize,
    xscale: f32,
    pe: std::sync::RwLock<Option<Tensor>>,
}

impl EspnetRelPositionalEncoding {
    pub fn new(d_model: usize) -> Result<Self> {
        Ok(Self {
            d_model,
            xscale: (d_model as f32).sqrt(),
            pe: std::sync::RwLock::new(None),
        })
    }

    fn extend_pe(&self, x: &Tensor) -> Result<()> {
        let seq_len = x.dim(1)?;
        
        // Fast path: use a read lock to check if we already have a long enough PE
        {
            let pe_read = self.pe.read().unwrap();
            if let Some(pe) = &*pe_read {
                if pe.dim(1)? >= seq_len * 2 - 1 {
                    return Ok(());
                }
            }
        }

        // Slow path: acquire a write lock and double check
        let mut pe_lock = self.pe.write().unwrap();
        if let Some(pe) = &*pe_lock {
            if pe.dim(1)? >= seq_len * 2 - 1 {
                return Ok(());
            }
        }

        // Suppose `i` means position of query vector and `j` means position of key vector.
        // We use relative positions.
        let mut pe_positive_data = vec![0.0f32; seq_len * self.d_model];
        let mut pe_negative_data = vec![0.0f32; seq_len * self.d_model];

        for pos in 0..seq_len {
            for i in (0..self.d_model).step_by(2) {
                let div_term = (-((i as f32) * (10000.0f32.ln() / (self.d_model as f32)))).exp();
                
                let val_pos = (pos as f32) * div_term;
                let val_neg = -(pos as f32) * div_term;

                pe_positive_data[pos * self.d_model + i] = val_pos.sin();
                pe_negative_data[pos * self.d_model + i] = val_neg.sin();

                if i + 1 < self.d_model {
                    pe_positive_data[pos * self.d_model + i + 1] = val_pos.cos();
                    pe_negative_data[pos * self.d_model + i + 1] = val_neg.cos();
                }
            }
        }

        let device = x.device();
        let pe_pos = Tensor::from_vec(pe_positive_data, (seq_len, self.d_model), device)?;
        let pe_neg = Tensor::from_vec(pe_negative_data, (seq_len, self.d_model), device)?;

        // Reverse the order of positive indices
        let mut pe_pos_rev_data = vec![0.0f32; seq_len * self.d_model];
        let pe_pos_vec = pe_pos.flatten_all()?.to_vec1::<f32>()?;
        for i in 0..seq_len {
            let rev_i = seq_len - 1 - i;
            for j in 0..self.d_model {
                pe_pos_rev_data[rev_i * self.d_model + j] = pe_pos_vec[i * self.d_model + j];
            }
        }
        let pe_pos_rev = Tensor::from_vec(pe_pos_rev_data, (1, seq_len, self.d_model), device)?;
        
        let pe_neg_slice = pe_neg.narrow(0, 1, seq_len - 1)?.unsqueeze(0)?;

        let pe = Tensor::cat(&[&pe_pos_rev, &pe_neg_slice], 1)?;
        *pe_lock = Some(pe);

        Ok(())
    }

    pub fn forward(&self, x: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        self.extend_pe(x)?;
        let x_scaled = x.affine(self.xscale as f64, 0.0)?;
        let pos_emb = self.position_encoding(offset, x.dim(1)?)?;
        Ok((x_scaled, pos_emb))
    }

    pub fn position_encoding(&self, _offset: usize, size: usize) -> Result<Tensor> {
        let pe_lock = self.pe.read().unwrap();
        let pe = pe_lock.as_ref().unwrap();
        let pe_len = pe.dim(1)?;
        let start = (pe_len / 2).saturating_sub(size).saturating_add(1);
        pe.narrow(1, start, size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_positional_encoding() -> Result<()> {
        let device = Device::Cpu;
        let d_model = 4;
        let max_len = 10;
        
        let pe = PositionalEncoding::new(d_model, max_len, false, &device)?;
        let x = Tensor::ones((2, 5, d_model), candle_core::DType::F32, &device)?;
        
        let (out, pos_emb) = pe.forward(&x, 0)?;
        
        assert_eq!(out.dims(), &[2, 5, d_model]);
        assert_eq!(pos_emb.dims(), &[1, 5, d_model]);
        
        Ok(())
    }

    #[test]
    fn test_espnet_rel_positional_encoding() -> Result<()> {
        let device = Device::Cpu;
        let d_model = 4;
        
        let pe = EspnetRelPositionalEncoding::new(d_model)?;
        let x = Tensor::ones((2, 5, d_model), candle_core::DType::F32, &device)?;
        
        let (out, pos_emb) = pe.forward(&x, 0)?;
        
        assert_eq!(out.dims(), &[2, 5, d_model]);
        assert_eq!(pos_emb.dims(), &[1, 5, d_model]);
        
        Ok(())
    }
}
