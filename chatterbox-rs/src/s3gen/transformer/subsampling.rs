use candle_core::{Result, Tensor, Module};
use candle_nn::{Linear, LayerNorm, VarBuilder, Conv1d, Conv1dConfig, conv1d, Conv2d, Conv2dConfig, conv2d, linear};
use super::embedding::{PositionalEncoding, RelPositionalEncoding, NoPositionalEncoding, EspnetRelPositionalEncoding};

// A trait for sub-sampling layers to implement
pub trait Subsampling: Send + Sync {
    fn forward(
        &self,
        x: &Tensor,
        x_mask: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)>;
    
    fn right_context(&self) -> usize;
    fn subsampling_rate(&self) -> usize;
}

// Subsampling requires positional encoding, so let's wrap PositionalEncoding structs into an enum.
pub enum PosEncEnum {
    PositionalEncoding(PositionalEncoding),
    RelPositionalEncoding(RelPositionalEncoding),
    NoPositionalEncoding(NoPositionalEncoding),
    EspnetRelPositionalEncoding(EspnetRelPositionalEncoding),
}

impl PosEncEnum {
    pub fn forward(&self, x: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        match self {
            PosEncEnum::PositionalEncoding(pe) => pe.forward(x, offset),
            PosEncEnum::RelPositionalEncoding(pe) => pe.forward(x, offset),
            PosEncEnum::NoPositionalEncoding(pe) => pe.forward(x, offset),
            PosEncEnum::EspnetRelPositionalEncoding(pe) => pe.forward(x, offset),
        }
    }
}

pub struct BaseSubsampling {
    pub right_context: usize,
    pub subsampling_rate: usize,
}

impl BaseSubsampling {
    pub fn right_context(&self) -> usize {
        self.right_context
    }
    
    pub fn subsampling_rate(&self) -> usize {
        self.subsampling_rate
    }
}

pub struct LinearNoSubsampling {
    out_linear: Linear,
    out_ln: LayerNorm,
    dropout_rate: f64,
    pos_enc: PosEncEnum,
    base: BaseSubsampling,
}

impl LinearNoSubsampling {
    pub fn new(
        idim: usize,
        odim: usize,
        dropout_rate: f64,
        pos_enc: PosEncEnum,
        vb: VarBuilder,
    ) -> Result<Self> {
        let out_linear = linear(idim, odim, vb.pp("out.0"))?;
        let out_ln = candle_nn::layer_norm(odim, 1e-5, vb.pp("out.1"))?;
        Ok(Self {
            out_linear,
            out_ln,
            dropout_rate,
            pos_enc,
            base: BaseSubsampling {
                right_context: 0,
                subsampling_rate: 1,
            },
        })
    }
}

impl Subsampling for LinearNoSubsampling {
    fn forward(
        &self,
        x: &Tensor,
        x_mask: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let x = self.out_linear.forward(x)?;
        let x = self.out_ln.forward(&x)?;
        
        let x = if self.dropout_rate > 0.0 {
            candle_nn::ops::dropout(&x, self.dropout_rate as f32)?
        } else {
            x
        };
        
        let x = x.relu()?;
        let (x, pos_emb) = self.pos_enc.forward(&x, offset)?;
        Ok((x, pos_emb, x_mask.clone()))
    }
    
    fn right_context(&self) -> usize {
        self.base.right_context
    }
    
    fn subsampling_rate(&self) -> usize {
        self.base.subsampling_rate
    }
}

pub struct Conv2dSubsampling4 {
    conv1: Conv2d,
    conv2: Conv2d,
    out_linear: Linear,
    pos_enc: PosEncEnum,
    base: BaseSubsampling,
}

impl Conv2dSubsampling4 {
    pub fn new(
        idim: usize,
        odim: usize,
        _dropout_rate: f64,
        pos_enc: PosEncEnum,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv1 = conv2d(1, odim, 3, Conv2dConfig { stride: 2, ..Default::default() }, vb.pp("conv.0"))?;
        let conv2 = conv2d(odim, odim, 3, Conv2dConfig { stride: 2, ..Default::default() }, vb.pp("conv.2"))?;
        
        let in_linear = odim * (((idim - 1) / 2 - 1) / 2);
        let out_linear = linear(in_linear, odim, vb.pp("out.0"))?;
        
        Ok(Self {
            conv1,
            conv2,
            out_linear,
            pos_enc,
            base: BaseSubsampling {
                right_context: 6,
                subsampling_rate: 4,
            },
        })
    }
}

impl Subsampling for Conv2dSubsampling4 {
    fn forward(
        &self,
        x: &Tensor,
        x_mask: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // x: (B, T, idim)
        let x = x.unsqueeze(1)?; // (B, 1, T, idim)
        let x = self.conv1.forward(&x)?.relu()?;
        let x = self.conv2.forward(&x)?.relu()?;
        
        let (b, c, t, f) = x.dims4()?;
        let x = x.transpose(1, 2)?.contiguous()?.reshape((b, t, c * f))?;
        let x = self.out_linear.forward(&x)?;
        let (x, pos_emb) = self.pos_enc.forward(&x, offset)?;
        
        // x_mask: (B, 1, T)
        // Subsampling the mask: x_mask[:, :, 2::2][:, :, 2::2]
        // This takes step 2 starting at 2, twice. Which means T/4 in length, starting at 4? Wait.
        // x_mask[:, :, 2::2] takes 2, 4, 6, 8...
        // Applying again takes 6, 10, 14...
        // Actually Candle doesn't have an easy slice for 2::2. We might just slice x_mask to the required length `t`.
        // We know the new time dimension is `t`.
        let mask_t = x_mask.dim(2)?;
        // Let's implement a step slice for x_mask using narrow and arange_step or just by picking indices.
        let indices = Tensor::arange_step(6u32, mask_t as u32, 4, x.device())?;
        
        // If indices has a different size than t, we must narrow it. (Often padding issues).
        // Let's just create a narrow directly or use index_select.
        let indices = if indices.elem_count() > t {
            indices.narrow(0, 0, t)?
        } else {
            indices
        };
        let x_mask = x_mask.index_select(&indices, 2)?;
        
        Ok((x, pos_emb, x_mask))
    }
    
    fn right_context(&self) -> usize {
        self.base.right_context
    }
    
    fn subsampling_rate(&self) -> usize {
        self.base.subsampling_rate
    }
}

// Similarly for Conv2dSubsampling6, Conv2dSubsampling8, Conv1dSubsampling2...

pub struct Conv1dSubsampling2 {
    conv1: Conv1d,
    conv2: Conv1d,
    pos_enc: PosEncEnum,
    base: BaseSubsampling,
}

impl Conv1dSubsampling2 {
    pub fn new(
        idim: usize,
        odim: usize,
        _dropout_rate: f64,
        pos_enc: PosEncEnum,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv1 = conv1d(idim, odim, 3, Conv1dConfig { padding: 1, ..Default::default() }, vb.pp("conv.0"))?;
        let conv2 = conv1d(odim, odim, 3, Conv1dConfig { padding: 1, stride: 2, ..Default::default() }, vb.pp("conv.2"))?;
        Ok(Self {
            conv1,
            conv2,
            pos_enc,
            base: BaseSubsampling {
                right_context: 4,
                subsampling_rate: 2,
            },
        })
    }
}

impl Subsampling for Conv1dSubsampling2 {
    fn forward(
        &self,
        x: &Tensor,
        x_mask: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let time = x.dim(1)?;
        let x = x.transpose(1, 2)?.contiguous()?; // (b, f, t)
        
        let x = self.conv1.forward(&x)?;
        let x = x.gelu()?;
        let x = self.conv2.forward(&x)?;
        let x = x.gelu()?;
        
        let x = x.transpose(1, 2)?.contiguous()?; // (b, t, f)
        let (x, pos_emb) = self.pos_enc.forward(&x, offset)?;
        
        let t = x.dim(1)?;
        let start = (time + 1) % 2;
        let indices = Tensor::arange_step(start as u32, x_mask.dim(2)? as u32, 2, x_mask.device())?;
        let indices = if indices.elem_count() > t {
            indices.narrow(0, 0, t)?
        } else {
            indices
        };
        let x_mask = x_mask.index_select(&indices, 2)?;
        
        Ok((x, pos_emb, x_mask))
    }
    
    fn right_context(&self) -> usize {
        self.base.right_context
    }
    
    fn subsampling_rate(&self) -> usize {
        self.base.subsampling_rate
    }
}

pub struct Conv2dSubsampling6 {
    conv1: Conv2d,
    conv2: Conv2d,
    linear: Linear,
    pos_enc: PosEncEnum,
    base: BaseSubsampling,
}

impl Conv2dSubsampling6 {
    pub fn new(
        idim: usize,
        odim: usize,
        _dropout_rate: f64,
        pos_enc: PosEncEnum,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv1 = conv2d(1, odim, 3, Conv2dConfig { stride: 2, ..Default::default() }, vb.pp("conv.0"))?;
        let conv2 = conv2d(odim, odim, 5, Conv2dConfig { stride: 3, ..Default::default() }, vb.pp("conv.2"))?;
        
        let in_linear = odim * (((idim - 1) / 2 - 2) / 3);
        let linear_layer = linear(in_linear, odim, vb.pp("linear"))?;
        
        Ok(Self {
            conv1,
            conv2,
            linear: linear_layer,
            pos_enc,
            base: BaseSubsampling {
                right_context: 10,
                subsampling_rate: 6,
            },
        })
    }
}

impl Subsampling for Conv2dSubsampling6 {
    fn forward(
        &self,
        x: &Tensor,
        x_mask: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let x = x.unsqueeze(1)?;
        let x = self.conv1.forward(&x)?.relu()?;
        let x = self.conv2.forward(&x)?.relu()?;
        
        let (b, c, t, f) = x.dims4()?;
        let x = x.transpose(1, 2)?.contiguous()?.reshape((b, t, c * f))?;
        let x = self.linear.forward(&x)?;
        let (x, pos_emb) = self.pos_enc.forward(&x, offset)?;
        
        // x_mask[:, :, 2::2][:, :, 4::3] -> starts at 2, steps by 2. Then starts at 4th element (index 4 of the new sequence which means index 10 in original) and steps by 3 (so 6 in original).
        let indices = Tensor::arange_step(10u32, x_mask.dim(2)? as u32, 6, x_mask.device())?;
        let indices = if indices.elem_count() > t {
            indices.narrow(0, 0, t)?
        } else {
            indices
        };
        let x_mask = x_mask.index_select(&indices, 2)?;
        
        Ok((x, pos_emb, x_mask))
    }
    
    fn right_context(&self) -> usize {
        self.base.right_context
    }
    
    fn subsampling_rate(&self) -> usize {
        self.base.subsampling_rate
    }
}

pub struct Conv2dSubsampling8 {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    linear: Linear,
    pos_enc: PosEncEnum,
    base: BaseSubsampling,
}

impl Conv2dSubsampling8 {
    pub fn new(
        idim: usize,
        odim: usize,
        _dropout_rate: f64,
        pos_enc: PosEncEnum,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv1 = conv2d(1, odim, 3, Conv2dConfig { stride: 2, ..Default::default() }, vb.pp("conv.0"))?;
        let conv2 = conv2d(odim, odim, 3, Conv2dConfig { stride: 2, ..Default::default() }, vb.pp("conv.2"))?;
        let conv3 = conv2d(odim, odim, 3, Conv2dConfig { stride: 2, ..Default::default() }, vb.pp("conv.4"))?;
        
        let in_linear = odim * ((((idim - 1) / 2 - 1) / 2 - 1) / 2);
        let linear_layer = linear(in_linear, odim, vb.pp("linear"))?;
        
        Ok(Self {
            conv1,
            conv2,
            conv3,
            linear: linear_layer,
            pos_enc,
            base: BaseSubsampling {
                right_context: 14,
                subsampling_rate: 8,
            },
        })
    }
}

impl Subsampling for Conv2dSubsampling8 {
    fn forward(
        &self,
        x: &Tensor,
        x_mask: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let x = x.unsqueeze(1)?;
        let x = self.conv1.forward(&x)?.relu()?;
        let x = self.conv2.forward(&x)?.relu()?;
        let x = self.conv3.forward(&x)?.relu()?;
        
        let (b, c, t, f) = x.dims4()?;
        let x = x.transpose(1, 2)?.contiguous()?.reshape((b, t, c * f))?;
        let x = self.linear.forward(&x)?;
        let (x, pos_emb) = self.pos_enc.forward(&x, offset)?;
        
        let indices = Tensor::arange_step(14u32, x_mask.dim(2)? as u32, 8, x_mask.device())?;
        let indices = if indices.elem_count() > t {
            indices.narrow(0, 0, t)?
        } else {
            indices
        };
        let x_mask = x_mask.index_select(&indices, 2)?;
        
        Ok((x, pos_emb, x_mask))
    }
    
    fn right_context(&self) -> usize {
        self.base.right_context
    }
    
    fn subsampling_rate(&self) -> usize {
        self.base.subsampling_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    use candle_nn::VarMap;

    fn get_no_pos_enc(dim: usize) -> Result<PosEncEnum> {
        Ok(PosEncEnum::NoPositionalEncoding(NoPositionalEncoding::new(dim)?))
    }

    #[test]
    fn test_linear_no_subsampling() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let idim = 80;
        let odim = 256;
        let sub = LinearNoSubsampling::new(idim, odim, 0.1, get_no_pos_enc(odim)?, vb)?;
        
        let xs = Tensor::randn(0.0f32, 1.0, (2, 10, idim), &device)?;
        let mask = Tensor::ones((2, 1, 10), DType::U8, &device)?;
        
        let (out, _pos_emb, out_mask) = sub.forward(&xs, &mask, 0)?;
        assert_eq!(out.dims3()?, (2, 10, odim));
        assert_eq!(out_mask.dims3()?, (2, 1, 10));
        assert_eq!(sub.subsampling_rate(), 1);
        Ok(())
    }

    #[test]
    fn test_conv2d_subsampling4() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let idim = 80; // (80-1)/2 = 39, (39-1)/2 = 19
        let odim = 256;
        let sub = Conv2dSubsampling4::new(idim, odim, 0.1, get_no_pos_enc(odim)?, vb)?;
        
        let xs = Tensor::randn(0.0f32, 1.0, (2, 20, idim), &device)?;
        let mask = Tensor::ones((2, 1, 20), DType::U8, &device)?;
        
        let (out, _pos_emb, _out_mask) = sub.forward(&xs, &mask, 0)?;
        // After 2 convs with stride 2: 20 -> 9 -> 4 roughly (depending on padding/dilation).
        // Let's print out dims to verify.
        let (_, t, _) = out.dims3()?;
        assert!(t < 20);
        assert_eq!(sub.subsampling_rate(), 4);
        Ok(())
    }
}
