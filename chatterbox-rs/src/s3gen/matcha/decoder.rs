use candle_core::{Result, Tensor, DType};
use candle_nn::{VarBuilder, Module, Conv1d, Conv1dConfig};
use candle_nn::group_norm::{GroupNorm, group_norm};

/// Sinusoidal positional embeddings
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
        let x = if x.dims().len() < 1 {
            x.unsqueeze(0)?
        } else {
            x.clone()
        };
        let device = x.device();
        let half_dim = self.dim / 2;
        let emb = (10000.0f64.ln() / (half_dim as f64 - 1.0)) as f32;

        // torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        let arange = Tensor::arange(0u32, half_dim as u32, device)?.to_dtype(DType::F32)?;
        let emb = (arange * (-emb as f64))?.exp()?;

        // emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        let emb = emb.unsqueeze(0)?;
        let x_unsqueezed = x.unsqueeze(1)?;
        let scaled_x = (x_unsqueezed * (scale as f64))?;
        let emb = scaled_x.broadcast_mul(&emb)?;

        // emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        let sin_emb = emb.sin()?;
        let cos_emb = emb.cos()?;
        let emb = Tensor::cat(&[&sin_emb, &cos_emb], candle_core::D::Minus1)?;

        Ok(emb)
    }
}

/// Block1D
pub struct Block1D {
    conv1: Conv1d,
    norm1: GroupNorm,
}

impl Block1D {
    pub fn new(dim_in: usize, dim_out: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv1dConfig { padding: 1, ..Default::default() };
        let conv1 = candle_nn::conv1d(dim_in, dim_out, 3, conv_cfg, vb.pp("block.0"))?;
        let norm1 = group_norm(groups, dim_out, 1e-5, vb.pp("block.1"))?;
        Ok(Self { conv1, norm1 })
    }

    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // x * mask
        let masked_x = x.broadcast_mul(mask)?;

        // conv1d
        let h = self.conv1.forward(&masked_x)?;

        // group norm
        let h = self.norm1.forward(&h)?;

        // mish: x * tanh(softplus(x)) where softplus(x) = ln(1 + e^x)
        // using x * (x.exp().affine(1,1).log().tanh())?
        let h_exp = h.exp()?;
        let softplus = (h_exp + 1.0)?.log()?;
        let tanh_softplus = softplus.tanh()?;
        let h = h.broadcast_mul(&tanh_softplus)?;

        // h * mask
        let h = h.broadcast_mul(mask)?;

        Ok(h)
    }
}

use candle_nn::{Linear, linear};
use super::transformer::BasicTransformerBlock;

pub struct ResnetBlock1D {
    mlp: Linear,
    block1: Block1D,
    block2: Block1D,
    res_conv: Conv1d,
}

impl ResnetBlock1D {
    pub fn new(dim: usize, dim_out: usize, time_emb_dim: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        let mlp = linear(time_emb_dim, dim_out, vb.pp("mlp.1"))?;
        let block1 = Block1D::new(dim, dim_out, groups, vb.pp("block1"))?;
        let block2 = Block1D::new(dim_out, dim_out, groups, vb.pp("block2"))?;
        let res_conv = candle_nn::conv1d(dim, dim_out, 1, Default::default(), vb.pp("res_conv"))?;

        Ok(Self { mlp, block1, block2, res_conv })
    }

    pub fn forward(&self, x: &Tensor, mask: &Tensor, time_emb: &Tensor) -> Result<Tensor> {
        let h = self.block1.forward(x, mask)?;

        // Mish then Linear for MLP
        let t_exp = time_emb.exp()?;
        let t_softplus = (t_exp + 1.0)?.log()?;
        let t_mish = time_emb.broadcast_mul(&t_softplus.tanh()?)?;

        let t_proj = self.mlp.forward(&t_mish)?;

        let h = h.broadcast_add(&t_proj.unsqueeze(candle_core::D::Minus1)?)?;
        let h = self.block2.forward(&h, mask)?;

        let masked_x = x.broadcast_mul(mask)?;
        let res = self.res_conv.forward(&masked_x)?;

        let output = (h + res)?;
        Ok(output)
    }
}

pub struct Downsample1D {
    conv: Conv1d,
}

impl Downsample1D {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv1dConfig { stride: 2, padding: 1, ..Default::default() };
        let conv = candle_nn::conv1d(dim, dim, 3, conv_cfg, vb.pp("conv"))?;
        Ok(Self { conv })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    pub fn new(in_channels: usize, time_embed_dim: usize, _act_fn: &str, vb: VarBuilder) -> Result<Self> {
        let linear_1 = linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;
        let linear_2 = linear(time_embed_dim, time_embed_dim, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, sample: &Tensor) -> Result<Tensor> {
        let sample = self.linear_1.forward(sample)?;
        // silu act
        let sample = candle_nn::ops::silu(&sample)?;
        let sample = self.linear_2.forward(&sample)?;
        Ok(sample)
    }
}

pub struct Upsample1D {
    channels: usize,
    use_conv_transpose: bool,
    conv_transpose: Option<candle_nn::ConvTranspose1d>,
    conv: Option<Conv1d>,
}

impl Upsample1D {
    pub fn new(channels: usize, use_conv_transpose: bool, vb: VarBuilder) -> Result<Self> {
        let out_channels = channels;
        let mut conv_transpose = None;
        let mut conv = None;

        if use_conv_transpose {
            let cfg = candle_nn::ConvTranspose1dConfig { stride: 2, padding: 1, ..Default::default() };
            conv_transpose = Some(candle_nn::conv_transpose1d(channels, out_channels, 4, cfg, vb.pp("conv"))?);
        } else {
            let cfg = Conv1dConfig { padding: 1, ..Default::default() };
            conv = Some(candle_nn::conv1d(channels, out_channels, 3, cfg, vb.pp("conv"))?);
        }

        Ok(Self { channels, use_conv_transpose, conv_transpose, conv })
    }

    pub fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        if self.use_conv_transpose {
            if let Some(ct) = &self.conv_transpose {
                return ct.forward(inputs);
            }
        }

        // F.interpolate(inputs, scale_factor=2.0, mode="nearest")
        let (b, c, l) = inputs.dims3()?;
        let outputs = inputs.unsqueeze(candle_core::D::Minus1)?;
        let outputs = outputs.broadcast_as((b, c, l, 2))?;
        let outputs = outputs.contiguous()?.reshape((b, c, l * 2))?;

        if let Some(c) = &self.conv {
            return c.forward(&outputs);
        }

        Ok(outputs)
    }
}

pub enum BlockWrapper {
    Transformer(BasicTransformerBlock),
    // Conformer wrapper will be implemented later when needed
    // Conformer(ConformerWrapper),
}

impl BlockWrapper {
    pub fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>, timestep: Option<&Tensor>) -> Result<Tensor> {
        match self {
            Self::Transformer(block) => block.forward(hidden_states, attention_mask, timestep),
        }
    }
}

pub struct DownBlock {
    resnet: ResnetBlock1D,
    transformer_blocks: Vec<BlockWrapper>,
    downsample: Option<Downsample1D>,
    downsample_conv: Option<Conv1d>,
}

impl DownBlock {
    pub fn forward(&self, x: &Tensor, mask_down: &Tensor, t: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = self.resnet.forward(x, mask_down, t)?;

        let mut x = x.transpose(1, 2)?; // b c t -> b t c
        let mut mask_t = mask_down.squeeze(1)?; // b 1 t -> b t

        for block in &self.transformer_blocks {
            x = block.forward(&x, Some(&mask_t), Some(t))?;
        }

        let x = x.transpose(1, 2)?; // b t c -> b c t
        let mask_down_new = mask_t.unsqueeze(1)?; // b t -> b 1 t

        let hidden = x.clone(); // save for skip connection

        let x_masked = x.broadcast_mul(&mask_down_new)?;

        let x = if let Some(downsample) = &self.downsample {
            downsample.forward(&x_masked)?
        } else if let Some(conv) = &self.downsample_conv {
            conv.forward(&x_masked)?
        } else {
            x_masked
        };

        Ok((x, hidden))
    }
}

pub struct MidBlock {
    resnet: ResnetBlock1D,
    transformer_blocks: Vec<BlockWrapper>,
}

impl MidBlock {
    pub fn forward(&self, x: &Tensor, mask_mid: &Tensor, t: &Tensor) -> Result<Tensor> {
        let x = self.resnet.forward(x, mask_mid, t)?;
        let mut x = x.transpose(1, 2)?; // b c t -> b t c
        let mask_t = mask_mid.squeeze(1)?; // b 1 t -> b t

        for block in &self.transformer_blocks {
            x = block.forward(&x, Some(&mask_t), Some(t))?;
        }

        x.transpose(1, 2) // b t c -> b c t
    }
}

pub struct UpBlock {
    resnet: ResnetBlock1D,
    transformer_blocks: Vec<BlockWrapper>,
    upsample: Option<Upsample1D>,
    upsample_conv: Option<Conv1d>,
}

impl UpBlock {
    pub fn forward(&self, x: &Tensor, hidden: &Tensor, mask_up: &Tensor, t: &Tensor) -> Result<Tensor> {
        // pack([x, hiddens.pop()], "b * t")[0]
        let x = Tensor::cat(&[x, hidden], 1)?;
        let x = self.resnet.forward(&x, mask_up, t)?;

        let mut x = x.transpose(1, 2)?; // b c t -> b t c
        let mask_t = mask_up.squeeze(1)?; // b 1 t -> b t

        for block in &self.transformer_blocks {
            x = block.forward(&x, Some(&mask_t), Some(t))?;
        }

        let x = x.transpose(1, 2)?; // b t c -> b c t
        let mask_up_new = mask_t.unsqueeze(1)?; // b t -> b 1 t

        let x_masked = x.broadcast_mul(&mask_up_new)?;

        let x = if let Some(upsample) = &self.upsample {
            upsample.forward(&x_masked)?
        } else if let Some(conv) = &self.upsample_conv {
            conv.forward(&x_masked)?
        } else {
            x_masked
        };

        Ok(x)
    }
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
    #[allow(clippy::too_many_arguments)]
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
        let time_mlp = TimestepEmbedding::new(in_channels, time_embed_dim, "silu", vb.pp("time_mlp"))?;

        let mut down_blocks = Vec::new();
        let mut mid_blocks = Vec::new();
        let mut up_blocks = Vec::new();

        let mut output_channel = in_channels;

        // Down blocks
        for (i, &ch) in channels.iter().enumerate() {
            let input_channel = output_channel;
            output_channel = ch;
            let is_last = i == channels.len() - 1;

            let resnet = ResnetBlock1D::new(input_channel, output_channel, time_embed_dim, 8, vb.pp(&format!("down_blocks.{}.0", i)))?;

            let mut transformer_blocks = Vec::new();
            for j in 0..n_blocks {
                let block = Self::get_block(
                    down_block_type,
                    output_channel,
                    attention_head_dim,
                    num_heads,
                    dropout,
                    act_fn,
                    vb.pp(&format!("down_blocks.{}.1.{}", i, j)),
                )?;
                transformer_blocks.push(block);
            }

            let mut downsample = None;
            let mut downsample_conv = None;

            if !is_last {
                downsample = Some(Downsample1D::new(output_channel, vb.pp(&format!("down_blocks.{}.2", i)))?);
            } else {
                let cfg = Conv1dConfig { padding: 1, ..Default::default() };
                downsample_conv = Some(candle_nn::conv1d(output_channel, output_channel, 3, cfg, vb.pp(&format!("down_blocks.{}.2", i)))?);
            }

            down_blocks.push(DownBlock { resnet, transformer_blocks, downsample, downsample_conv });
        }

        // Mid blocks
        let input_channel = channels[channels.len() - 1];
        let output_channel = channels[channels.len() - 1];

        for i in 0..num_mid_blocks {
            let resnet = ResnetBlock1D::new(input_channel, output_channel, time_embed_dim, 8, vb.pp(&format!("mid_blocks.{}.0", i)))?;

            let mut transformer_blocks = Vec::new();
            for j in 0..n_blocks {
                let block = Self::get_block(
                    mid_block_type,
                    output_channel,
                    attention_head_dim,
                    num_heads,
                    dropout,
                    act_fn,
                    vb.pp(&format!("mid_blocks.{}.1.{}", i, j)),
                )?;
                transformer_blocks.push(block);
            }

            mid_blocks.push(MidBlock { resnet, transformer_blocks });
        }

        // Up blocks
        let mut rev_channels = channels.to_vec();
        rev_channels.reverse();
        rev_channels.push(channels[0]);

        for i in 0..(rev_channels.len() - 1) {
            let input_channel = rev_channels[i];
            let output_channel = rev_channels[i + 1];
            let is_last = i == rev_channels.len() - 2;

            let resnet = ResnetBlock1D::new(2 * input_channel, output_channel, time_embed_dim, 8, vb.pp(&format!("up_blocks.{}.0", i)))?;

            let mut transformer_blocks = Vec::new();
            for j in 0..n_blocks {
                let block = Self::get_block(
                    up_block_type,
                    output_channel,
                    attention_head_dim,
                    num_heads,
                    dropout,
                    act_fn,
                    vb.pp(&format!("up_blocks.{}.1.{}", i, j)),
                )?;
                transformer_blocks.push(block);
            }

            let mut upsample = None;
            let mut upsample_conv = None;

            if !is_last {
                upsample = Some(Upsample1D::new(output_channel, true, vb.pp(&format!("up_blocks.{}.2", i)))?);
            } else {
                let cfg = Conv1dConfig { padding: 1, ..Default::default() };
                upsample_conv = Some(candle_nn::conv1d(output_channel, output_channel, 3, cfg, vb.pp(&format!("up_blocks.{}.2", i)))?);
            }

            up_blocks.push(UpBlock { resnet, transformer_blocks, upsample, upsample_conv });
        }

        let final_ch = channels[0];
        let final_block = Block1D::new(final_ch, final_ch, 8, vb.pp("final_block"))?;
        let final_proj = candle_nn::conv1d(final_ch, out_channels, 1, Default::default(), vb.pp("final_proj"))?;

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

    fn get_block(
        block_type: &str,
        dim: usize,
        attention_head_dim: usize,
        num_heads: usize,
        dropout: f64,
        act_fn: &str,
        vb: VarBuilder,
    ) -> Result<BlockWrapper> {
        if block_type == "transformer" {
            let block = BasicTransformerBlock::new(dim, num_heads, attention_head_dim, dropout, act_fn, vb)?;
            Ok(BlockWrapper::Transformer(block))
        } else {
            candle_core::bail!("Unknown block type {}", block_type);
        }
    }

    pub fn forward(&self, x: &Tensor, mask: &Tensor, mu: &Tensor, t: &Tensor, spks: Option<&Tensor>, cond: Option<&Tensor>) -> Result<Tensor> {
        let t_emb = self.time_embeddings.forward(t, 1000.0)?;
        let t_emb = self.time_mlp.forward(&t_emb)?;

        let mut x = Tensor::cat(&[x, mu], 1)?;

        if let Some(spks) = spks {
            // spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            let b = spks.dim(0)?;
            let c = spks.dim(1)?;
            let t_len = x.dim(2)?;
            let spks_repeated = spks.unsqueeze(2)?.broadcast_as((b, c, t_len))?;
            x = Tensor::cat(&[&x, &spks_repeated], 1)?;
        }

        let mut hiddens = Vec::new();
        let mut masks = vec![mask.clone()];

        for down_block in &self.down_blocks {
            let mask_down = masks.last().unwrap();
            let (x_new, hidden) = down_block.forward(&x, mask_down, &t_emb)?;
            x = x_new;
            hiddens.push(hidden);

            // masks.append(mask_down[:, :, ::2])
            // Simplified sub-sampling for exactly factor 2
            let (b, c, l) = mask_down.dims3()?;
            if l % 2 == 0 {
                let m = mask_down.reshape((b, c, l / 2, 2))?;
                let m = m.narrow(3, 0, 1)?.reshape((b, c, l / 2))?;
                masks.push(m);
            } else {
                // Approximate fallback for odd length
                let idxs: Vec<u32> = (0..(l as u32)).step_by(2).collect();
                let idxs = Tensor::new(idxs.as_slice(), mask_down.device())?;
                let m = mask_down.index_select(&idxs, 2)?;
                masks.push(m);
            }
        }

        masks.pop(); // masks = masks[:-1]
        let mut mask_mid = masks.last().unwrap().clone();

        for mid_block in &self.mid_blocks {
            x = mid_block.forward(&x, &mask_mid, &t_emb)?;
        }

        for up_block in &self.up_blocks {
            let mask_up = masks.pop().unwrap();
            let hidden = hiddens.pop().unwrap();

            x = up_block.forward(&x, &hidden, &mask_up, &t_emb)?;
        }

        let mask_up = mask_mid; // Using last known mask from pop

        let x = self.final_block.forward(&x, mask)?; // Need actual original mask here
        let x_masked = x.broadcast_mul(mask)?;

        let output = self.final_proj.forward(&x_masked)?;
        let output = output.broadcast_mul(mask)?;

        Ok(output)
    }
}
