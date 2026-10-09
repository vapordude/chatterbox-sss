use candle_core::{DType, Device, Module, Result, Tensor};
use crate::s3gen::transformer::encoder_layer::ConformerEncoderLayer;
use candle_nn::{conv1d, conv_transpose1d, group_norm, linear, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, GroupNorm, Linear, VarBuilder};

use super::transformer::BasicTransformerBlock;
// We mock ConformerEncoderLayer, we might just mock ConformerWrapper or skip it if BasicTransformerBlock is used.

pub struct SinusoidalPosEmb {
    dim: usize,
}

impl SinusoidalPosEmb {
    pub fn new(dim: usize) -> Result<Self> {
        if dim % 2 != 0 {
            candle_core::bail!("SinusoidalPosEmb requires dim to be even");
        }
        Ok(Self { dim })
    }

    pub fn forward(&self, x: &Tensor, scale: f32) -> Result<Tensor> {
        let mut x = x.clone();
        if x.rank() < 1 {
            x = x.unsqueeze(0)?;
        }
        let device = x.device();
        let half_dim = self.dim / 2;
        let emb = (10000.0_f32).ln() / ((half_dim as f32) - 1.0);

        let arange = Tensor::arange(0.0_f32, half_dim as f32, device)?;
        let emb_scalar = Tensor::new(&[-emb], device)?.broadcast_as(arange.dims())?;
        let emb_tensor = (arange * emb_scalar)?.exp()?;

        let scale_tensor = Tensor::new(&[scale], device)?.broadcast_as(x.unsqueeze(1)?.dims())?;
        let scaled_x = (x.unsqueeze(1)? * scale_tensor)?;
        let emb = scaled_x.broadcast_mul(&emb_tensor.unsqueeze(0)?)?;

        let emb_sin = emb.sin()?;
        let emb_cos = emb.cos()?;
        Tensor::cat(&[&emb_sin, &emb_cos], 1)
    }
}

// Mish activation
pub struct Mish;
impl Mish {
    pub fn new() -> Self { Self }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        // From memory: x.broadcast_mul(&x.exp()?.affine(1.0, 1.0)?.log()?.tanh()?)?
        x.broadcast_mul(&x.exp()?.affine(1.0, 1.0)?.log()?.tanh()?)
    }
}

pub struct Block1D {
    conv: Conv1d,
    group_norm: GroupNorm,
    mish: Mish,
}

impl Block1D {
    pub fn new(dim: usize, dim_out: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv1dConfig { padding: 1, ..Default::default() };
        let conv = conv1d(dim, dim_out, 3, conv_cfg, vb.pp("block.0"))?;
        let group_norm = group_norm(groups, dim_out, 1e-5, vb.pp("block.1"))?;
        Ok(Self { conv, group_norm, mish: Mish::new() })
    }

    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let x_masked = x.broadcast_mul(mask)?;
        let mut out = self.conv.forward(&x_masked)?;
        out = self.group_norm.forward(&out)?;
        out = self.mish.forward(&out)?;
        out.broadcast_mul(mask)
    }
}

pub struct ResnetBlock1D {
    mlp_mish: Mish,
    mlp_linear: Linear,
    block1: Block1D,
    block2: Block1D,
    res_conv: Conv1d,
}

impl ResnetBlock1D {
    pub fn new(dim: usize, dim_out: usize, time_emb_dim: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        let mlp_linear = linear(time_emb_dim, dim_out, vb.pp("mlp.1"))?;
        let block1 = Block1D::new(dim, dim_out, groups, vb.pp("block1"))?;
        let block2 = Block1D::new(dim_out, dim_out, groups, vb.pp("block2"))?;
        let res_conv = conv1d(dim, dim_out, 1, Default::default(), vb.pp("res_conv"))?;
        Ok(Self {
            mlp_mish: Mish::new(),
            mlp_linear,
            block1,
            block2,
            res_conv,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: &Tensor, time_emb: &Tensor) -> Result<Tensor> {
        let mut h = self.block1.forward(x, mask)?;
        let time_emb_act = self.mlp_mish.forward(time_emb)?;
        let time_emb_proj = self.mlp_linear.forward(&time_emb_act)?.unsqueeze(2)?; // .unsqueeze(-1)
        h = h.broadcast_add(&time_emb_proj)?;
        h = self.block2.forward(&h, mask)?;
        let res = self.res_conv.forward(&x.broadcast_mul(mask)?)?;
        h.broadcast_add(&res)
    }
}

pub struct Downsample1D {
    conv: Conv1d,
}

impl Downsample1D {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv1dConfig { padding: 1, stride: 2, ..Default::default() };
        let conv = conv1d(dim, dim, 3, conv_cfg, vb.pp("conv"))?;
        Ok(Self { conv })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

// SiLU activation function
pub struct Silu;
impl Silu {
    pub fn new() -> Self { Self }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::silu(x)
    }
}

pub struct TimestepEmbedding {
    linear_1: Linear,
    cond_proj: Option<Linear>,
    act: Silu, // Diffusers "silu"
    linear_2: Linear,
    // post_act is optional in diffusers but generally None here
}

impl TimestepEmbedding {
    pub fn new(in_channels: usize, time_embed_dim: usize, act_fn: &str, out_dim: Option<usize>, cond_proj_dim: Option<usize>, vb: VarBuilder) -> Result<Self> {
        if act_fn != "silu" {
            candle_core::bail!("Only silu is currently supported for TimestepEmbedding act_fn");
        }
        let linear_1 = linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;
        let cond_proj = if let Some(c_dim) = cond_proj_dim {
            Some(candle_nn::linear_no_bias(c_dim, in_channels, vb.pp("cond_proj"))?)
        } else {
            None
        };

        let time_embed_dim_out = out_dim.unwrap_or(time_embed_dim);
        let linear_2 = linear(time_embed_dim, time_embed_dim_out, vb.pp("linear_2"))?;

        Ok(Self {
            linear_1,
            cond_proj,
            act: Silu::new(),
            linear_2,
        })
    }

    pub fn forward(&self, sample: &Tensor, condition: Option<&Tensor>) -> Result<Tensor> {
        let mut s = sample.clone();
        if let (Some(cond), Some(proj)) = (condition, &self.cond_proj) {
            s = s.broadcast_add(&proj.forward(cond)?)?;
        }
        s = self.linear_1.forward(&s)?;
        s = self.act.forward(&s)?;
        s = self.linear_2.forward(&s)?;
        Ok(s)
    }
}

pub enum UpsampleConv {
    ConvTranspose1d(ConvTranspose1d),
    Conv1d(Conv1d),
    None,
}

pub struct Upsample1D {
    channels: usize,
    out_channels: usize,
    use_conv: bool,
    use_conv_transpose: bool,
    conv: UpsampleConv,
}

impl Upsample1D {
    pub fn new(channels: usize, use_conv: bool, use_conv_transpose: bool, out_channels: Option<usize>, vb: VarBuilder) -> Result<Self> {
        let out_c = out_channels.unwrap_or(channels);
        let conv = if use_conv_transpose {
            let cfg = ConvTranspose1dConfig {
                padding: 1,
                stride: 2,
                ..Default::default()
            };
            UpsampleConv::ConvTranspose1d(conv_transpose1d(channels, out_c, 4, cfg, vb.pp("conv"))?)
        } else if use_conv {
            let cfg = Conv1dConfig { padding: 1, ..Default::default() };
            UpsampleConv::Conv1d(conv1d(channels, out_c, 3, cfg, vb.pp("conv"))?)
        } else {
            UpsampleConv::None
        };

        Ok(Self {
            channels,
            out_channels: out_c,
            use_conv,
            use_conv_transpose,
            conv,
        })
    }

    pub fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        if inputs.dim(1)? != self.channels {
            candle_core::bail!("Upsample1D expected {} channels, got {}", self.channels, inputs.dim(1)?);
        }

        if self.use_conv_transpose {
            if let UpsampleConv::ConvTranspose1d(ref c) = self.conv {
                return c.forward(inputs);
            }
        }

        // Nearest neighbor interpolation (scale_factor=2.0)
        let (b, c, l) = inputs.dims3()?;
        let outputs = inputs.unsqueeze(3)?.broadcast_as((b, c, l, 2))?.contiguous()?.reshape((b, c, l * 2))?;

        if self.use_conv {
            if let UpsampleConv::Conv1d(ref c) = self.conv {
                return c.forward(&outputs);
            }
        }

        Ok(outputs)
    }
}

// Mocks for blocks since full Decoder integrates them
pub enum DecoderBlock {
    ConformerWrapper(), // Full generic required or simple mock
    BasicTransformerBlock(BasicTransformerBlock),
}

impl DecoderBlock {
    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor, timestep: &Tensor) -> Result<Tensor> {
        match self {
            Self::BasicTransformerBlock(b) => b.forward(hidden_states, Some(attention_mask), Some(timestep)),
            Self::ConformerWrapper() => candle_core::bail!("ConformerWrapper forward not implemented yet"), // For tests we just use transformer
        }
    }
}

// Decoder Down/Mid/Up block bundles
pub struct DownBlock {
    resnet: ResnetBlock1D,
    transformer_blocks: Vec<DecoderBlock>,
    downsample: Option<Conv1d>,
    downsample_1d: Option<Downsample1D>,
}

pub struct MidBlock {
    resnet: ResnetBlock1D,
    transformer_blocks: Vec<DecoderBlock>,
}

pub struct UpBlock {
    resnet: ResnetBlock1D,
    transformer_blocks: Vec<DecoderBlock>,
    upsample: Option<Conv1d>,
    upsample_1d: Option<Upsample1D>,
}

pub struct Decoder {
    in_channels: usize,
    out_channels: usize,
    time_embeddings: SinusoidalPosEmb,
    time_mlp: TimestepEmbedding,
    down_blocks: Vec<DownBlock>,
    mid_blocks: Vec<MidBlock>,
    up_blocks: Vec<UpBlock>,
    final_block: Block1D,
    final_proj: Conv1d,
}

impl Decoder {
    fn get_block(block_type: &str, dim: usize, attention_head_dim: usize, num_heads: usize, dropout: f64, act_fn: &str, vb: VarBuilder) -> Result<DecoderBlock> {
        if block_type == "transformer" {
            Ok(DecoderBlock::BasicTransformerBlock(BasicTransformerBlock::new(
                dim,
                num_heads,
                attention_head_dim,
                dropout,
                act_fn,
                vb
            )?))
        } else {
            candle_core::bail!("Unknown block type {}", block_type);
        }
    }

    pub fn new(
        in_channels: usize,
        out_channels: usize,
        channels: &[usize],
        dropout: f64,
        attention_head_dim: usize,
        n_blocks: usize,
        num_mid_blocks: usize,
        num_heads: usize,
        act_fn: &str,
        down_block_type: &str,
        mid_block_type: &str,
        up_block_type: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let time_embeddings = SinusoidalPosEmb::new(in_channels)?;
        let time_embed_dim = channels[0] * 4;
        let time_mlp = TimestepEmbedding::new(in_channels, time_embed_dim, "silu", None, None, vb.pp("time_mlp"))?;

        let mut down_blocks = Vec::new();
        let mut output_channel = in_channels;
        for (i, &ch) in channels.iter().enumerate() {
            let input_channel = output_channel;
            output_channel = ch;
            let is_last = i == channels.len() - 1;

            let block_vb = vb.pp(&format!("down_blocks.{}", i));

            let resnet = ResnetBlock1D::new(input_channel, output_channel, time_embed_dim, 8, block_vb.pp("0"))?;

            let mut transformer_blocks = Vec::new();
            let tb_vb = block_vb.pp("1");
            for j in 0..n_blocks {
                transformer_blocks.push(Self::get_block(down_block_type, output_channel, attention_head_dim, num_heads, dropout, act_fn, tb_vb.pp(&j.to_string()))?);
            }

            let mut downsample = None;
            let mut downsample_1d = None;
            let ds_vb = block_vb.pp("2");
            if !is_last {
                downsample_1d = Some(Downsample1D::new(output_channel, ds_vb)?);
            } else {
                let cfg = Conv1dConfig { padding: 1, ..Default::default() };
                downsample = Some(conv1d(output_channel, output_channel, 3, cfg, ds_vb)?);
            }

            down_blocks.push(DownBlock {
                resnet,
                transformer_blocks,
                downsample,
                downsample_1d,
            });
        }

        let mut mid_blocks = Vec::new();
        for i in 0..num_mid_blocks {
            let input_channel = channels.last().cloned().unwrap_or(0);

            let block_vb = vb.pp(&format!("mid_blocks.{}", i));
            let resnet = ResnetBlock1D::new(input_channel, output_channel, time_embed_dim, 8, block_vb.pp("0"))?;

            let mut transformer_blocks = Vec::new();
            let tb_vb = block_vb.pp("1");
            for j in 0..n_blocks {
                transformer_blocks.push(Self::get_block(mid_block_type, output_channel, attention_head_dim, num_heads, dropout, act_fn, tb_vb.pp(&j.to_string()))?);
            }

            mid_blocks.push(MidBlock {
                resnet,
                transformer_blocks,
            });
        }

        let mut rev_channels = channels.to_vec();
        rev_channels.reverse();
        rev_channels.push(channels[0]);

        let mut up_blocks = Vec::new();
        for i in 0..(rev_channels.len() - 1) {
            let input_channel = rev_channels[i];
            let output_channel_up = rev_channels[i + 1];
            let is_last = i == rev_channels.len() - 2;

            let block_vb = vb.pp(&format!("up_blocks.{}", i));
            let resnet = ResnetBlock1D::new(2 * input_channel, output_channel_up, time_embed_dim, 8, block_vb.pp("0"))?;

            let mut transformer_blocks = Vec::new();
            let tb_vb = block_vb.pp("1");
            for j in 0..n_blocks {
                transformer_blocks.push(Self::get_block(up_block_type, output_channel_up, attention_head_dim, num_heads, dropout, act_fn, tb_vb.pp(&j.to_string()))?);
            }

            let mut upsample = None;
            let mut upsample_1d = None;
            let us_vb = block_vb.pp("2");
            if !is_last {
                upsample_1d = Some(Upsample1D::new(output_channel_up, false, true, None, us_vb)?);
            } else {
                let cfg = Conv1dConfig { padding: 1, ..Default::default() };
                upsample = Some(conv1d(output_channel_up, output_channel_up, 3, cfg, us_vb)?);
            }

            up_blocks.push(UpBlock {
                resnet,
                transformer_blocks,
                upsample,
                upsample_1d,
            });
        }

        let last_ch = rev_channels.last().cloned().unwrap_or(0);
        let final_block = Block1D::new(last_ch, last_ch, 8, vb.pp("final_block"))?;
        let final_proj = conv1d(last_ch, out_channels, 1, Default::default(), vb.pp("final_proj"))?;

        Ok(Self {
            in_channels,
            out_channels,
            time_embeddings,
            time_mlp,
            down_blocks,
            mid_blocks,
            up_blocks,
            final_block,
            final_proj,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_sinusoidal_pos_emb() -> Result<()> {
        let device = Device::Cpu;
        let pos_emb = SinusoidalPosEmb::new(256)?;

        let t = Tensor::new(&[1.0_f32, 2.0, 3.0, 4.0], &device)?;
        let out = pos_emb.forward(&t, 1000.0)?;

        assert_eq!(out.dims(), &[4, 256]);
        Ok(())
    }

    #[test]
    fn test_timestep_embedding() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let te = TimestepEmbedding::new(256, 1024, "silu", None, None, vb)?;

        let sample = Tensor::randn(0.0_f32, 1.0, (4, 256), &device)?;
        let out = te.forward(&sample, None)?;

        assert_eq!(out.dims(), &[4, 1024]);
        Ok(())
    }
}

impl DownBlock {
    pub fn forward(&self, mut x: Tensor, mask_down: &Tensor, t: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        x = self.resnet.forward(&x, mask_down, t)?;
        let mut x_t_c = x.transpose(1, 2)?.contiguous()?; // b c t -> b t c
        let mask_down_1d = mask_down.squeeze(1)?; // b 1 t -> b t

        for block in &self.transformer_blocks {
            x_t_c = block.forward(&x_t_c, &mask_down_1d, t)?;
        }
        x = x_t_c.transpose(1, 2)?.contiguous()?; // b t c -> b c t
        let hidden = x.clone();

        if let Some(ds) = &self.downsample_1d {
            x = ds.forward(&x.broadcast_mul(mask_down)?)?;
        } else if let Some(ds) = &self.downsample {
            x = ds.forward(&x.broadcast_mul(mask_down)?)?;
        }

        // masks.append(mask_down[:, :, ::2])
        let seq_len = mask_down.dim(2)?;
        let indices = Tensor::arange_step(0u32, seq_len as u32, 2, mask_down.device())?;
        let next_mask = mask_down.index_select(&indices, 2)?;

        Ok((x, hidden, next_mask))
    }
}

impl MidBlock {
    pub fn forward(&self, mut x: Tensor, mask_mid: &Tensor, t: &Tensor) -> Result<Tensor> {
        x = self.resnet.forward(&x, mask_mid, t)?;
        let mut x_t_c = x.transpose(1, 2)?.contiguous()?; // b c t -> b t c
        let mask_mid_1d = mask_mid.squeeze(1)?; // b 1 t -> b t

        for block in &self.transformer_blocks {
            x_t_c = block.forward(&x_t_c, &mask_mid_1d, t)?;
        }
        x_t_c.transpose(1, 2)?.contiguous() // b t c -> b c t
    }
}

impl UpBlock {
    pub fn forward(&self, x: Tensor, hidden: Tensor, mask_up: &Tensor, t: &Tensor) -> Result<Tensor> {
        let x_cat = Tensor::cat(&[&x, &hidden], 1)?; // pack([x, hiddens.pop()], "b * t")[0]
        let mut x = self.resnet.forward(&x_cat, mask_up, t)?;
        let mut x_t_c = x.transpose(1, 2)?.contiguous()?; // b c t -> b t c
        let mask_up_1d = mask_up.squeeze(1)?; // b 1 t -> b t

        for block in &self.transformer_blocks {
            x_t_c = block.forward(&x_t_c, &mask_up_1d, t)?;
        }
        x = x_t_c.transpose(1, 2)?.contiguous()?; // b t c -> b c t

        if let Some(us) = &self.upsample_1d {
            x = us.forward(&x.broadcast_mul(mask_up)?)?;
        } else if let Some(us) = &self.upsample {
            x = us.forward(&x.broadcast_mul(mask_up)?)?;
        }
        Ok(x)
    }
}

impl Decoder {
    pub fn forward(&self, x: &Tensor, mask: &Tensor, mu: &Tensor, t: &Tensor, spks: Option<&Tensor>, cond: Option<&Tensor>) -> Result<Tensor> {
        let t_emb = self.time_embeddings.forward(t, 1000.0)?.to_dtype(t.dtype())?;
        let t_emb = self.time_mlp.forward(&t_emb, None)?;

        let mut x = Tensor::cat(&[x, mu], 1)?;

        if let Some(spk) = spks {
            let b = x.dim(0)?;
            let c = spk.dim(1)?;
            let seq = x.dim(2)?;
            let spk_rep = spk.unsqueeze(2)?.broadcast_as((b, c, seq))?;
            x = Tensor::cat(&[&x, &spk_rep], 1)?;
        }

        let mut hiddens = Vec::new();
        let mut masks = vec![mask.clone()];

        for block in &self.down_blocks {
            let mask_down = masks.last().unwrap();
            let (out_x, hidden, next_mask) = block.forward(x, mask_down, &t_emb)?;
            x = out_x;
            hiddens.push(hidden);
            masks.push(next_mask);
        }

        masks.pop();
        let mask_mid = masks.last().unwrap();

        for block in &self.mid_blocks {
            x = block.forward(x, mask_mid, &t_emb)?;
        }

        for block in &self.up_blocks {
            let mask_up = masks.pop().unwrap();
            let hidden = hiddens.pop().unwrap();
            x = block.forward(x, hidden, &mask_up, &t_emb)?;
        }

        // Final proj uses mask_up from last iteration, but since it's popped it's not bound nicely in Rust if out of scope.
        // Actually, looking at Python: `mask_up` is the last one popped in the up_blocks loop. So it's equivalent to `mask`.
        // Let's use `mask` directly for final projection since upsampling restores sequence length.
        x = self.final_block.forward(&x, mask)?;
        let output = self.final_proj.forward(&x.broadcast_mul(mask)?)?;

        output.broadcast_mul(mask)
    }
}
