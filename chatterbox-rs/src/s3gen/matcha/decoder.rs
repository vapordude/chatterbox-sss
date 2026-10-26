use candle_core::{Tensor, Result, DType, Device};
use candle_nn::{Conv1d, GroupNorm, VarBuilder, Module};

/// Sinusoidal Positional Embedding
#[derive(Debug, Clone)]
pub struct SinusoidalPosEmb {
    /// The embedding dimension
    pub dim: usize,
}

impl SinusoidalPosEmb {
    /// Creates a new `SinusoidalPosEmb`
    pub fn new(dim: usize) -> Result<Self> {
        if dim % 2 != 0 {
            candle_core::bail!("SinusoidalPosEmb requires dim to be even");
        }
        Ok(Self { dim })
    }

    /// Forward pass for sinusoidal positional embedding
    pub fn forward(&self, x: &Tensor, scale: f64) -> Result<Tensor> {
        let x = if x.dims().is_empty() {
            x.unsqueeze(0)?
        } else {
            x.clone()
        };
        let device = x.device();
        let half_dim = self.dim / 2;
        let emb = (10000.0_f64).ln() / (half_dim as f64 - 1.0);

        let arange: Vec<f64> = (0..half_dim).map(|i| i as f64).collect();
        let arange = Tensor::from_vec(arange, (half_dim,), device)?.to_dtype(DType::F32)?;

        let emb_tensor = arange.affine(-emb, 0.0)?.exp()?;

        let x_f32 = x.to_dtype(DType::F32)?;
        let x_unsqueezed = x_f32.unsqueeze(x_f32.dims().len())?;
        let emb_unsqueezed = emb_tensor.unsqueeze(0)?;

        let x_scaled = x_unsqueezed.affine(scale, 0.0)?;
        let emb = x_scaled.broadcast_mul(&emb_unsqueezed)?;

        let sin_emb = emb.sin()?;
        let cos_emb = emb.cos()?;

        Tensor::cat(&[&sin_emb, &cos_emb], candle_core::D::Minus1)
    }
}

/// A 1D Convolutional Block with Group Normalization and Mish activation
#[derive(Debug, Clone)]
pub struct Block1D {
    /// 1D Convolution layer
    pub conv: Conv1d,
    /// Group Normalization layer
    pub norm: GroupNorm,
}

impl Block1D {
    /// Creates a new `Block1D` layer
    pub fn new(dim: usize, dim_out: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        let config = candle_nn::Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        // In python: Sequential(Conv1d, GroupNorm, Mish)
        // Paths might be "block.0" for conv, "block.1" for norm
        let conv = candle_nn::conv1d(dim, dim_out, 3, config, vb.pp("block.0"))?;
        let norm = candle_nn::group_norm(groups, dim_out, 1e-5, vb.pp("block.1"))?;

        Ok(Self { conv, norm })
    }

    /// Forward pass for `Block1D`
    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let mut out = x.broadcast_mul(mask)?;
        out = self.conv.forward(&out)?;
        out = self.norm.forward(&out)?;

        // Mish activation: x * tanh(ln(1 + e^x))
        out = out.broadcast_mul(&out.exp()?.affine(1.0, 1.0)?.log()?.tanh()?)?;

        out.broadcast_mul(mask)
    }
}

/// ResnetBlock1D module
#[derive(Debug, Clone)]
pub struct ResnetBlock1D {
    /// Linear layer for time embedding
    pub mlp_linear: candle_nn::Linear,
    /// First Block1D
    pub block1: Block1D,
    /// Second Block1D
    pub block2: Block1D,
    /// Residual 1D Convolution
    pub res_conv: Conv1d,
}

impl ResnetBlock1D {
    /// Creates a new `ResnetBlock1D` layer
    pub fn new(dim: usize, dim_out: usize, time_emb_dim: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        let mlp_linear = candle_nn::linear(time_emb_dim, dim_out, vb.pp("mlp.1"))?;
        let block1 = Block1D::new(dim, dim_out, groups, vb.pp("block1"))?;
        let block2 = Block1D::new(dim_out, dim_out, groups, vb.pp("block2"))?;

        let config = candle_nn::Conv1dConfig {
            padding: 0,
            ..Default::default()
        };
        let res_conv = candle_nn::conv1d(dim, dim_out, 1, config, vb.pp("res_conv"))?;

        Ok(Self {
            mlp_linear,
            block1,
            block2,
            res_conv,
        })
    }

    /// Forward pass for `ResnetBlock1D`
    pub fn forward(&self, x: &Tensor, mask: &Tensor, time_emb: &Tensor) -> Result<Tensor> {
        let mut h = self.block1.forward(x, mask)?;

        // Mish activation before linear for time_emb
        let time_emb_mish = time_emb.broadcast_mul(&time_emb.exp()?.affine(1.0, 1.0)?.log()?.tanh()?)?;
        let time_emb_proj = self.mlp_linear.forward(&time_emb_mish)?;
        let time_emb_unsqueezed = time_emb_proj.unsqueeze(time_emb_proj.dims().len())?;

        h = h.broadcast_add(&time_emb_unsqueezed)?;
        h = self.block2.forward(&h, mask)?;

        let res = self.res_conv.forward(&x.broadcast_mul(mask)?)?;
        h.broadcast_add(&res)
    }
}

/// Downsample1D module
#[derive(Debug, Clone)]
pub struct Downsample1D {
    /// 1D Convolution for downsampling
    pub conv: Conv1d,
}

impl Downsample1D {
    /// Creates a new `Downsample1D` layer
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let config = candle_nn::Conv1dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let conv = candle_nn::conv1d(dim, dim, 3, config, vb.pp("conv"))?;
        Ok(Self { conv })
    }

    /// Forward pass for `Downsample1D`
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

/// TimestepEmbedding module
#[derive(Debug, Clone)]
pub struct TimestepEmbedding {
    /// First linear layer
    pub linear_1: candle_nn::Linear,
    /// Second linear layer
    pub linear_2: candle_nn::Linear,
    /// Optional conditional projection layer
    pub cond_proj: Option<candle_nn::Linear>,
    /// Activation function type (default SiLU)
    pub act_fn: String,
}

impl TimestepEmbedding {
    /// Creates a new `TimestepEmbedding` layer
    pub fn new(
        in_channels: usize,
        time_embed_dim: usize,
        out_dim: Option<usize>,
        cond_proj_dim: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let linear_1 = candle_nn::linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;

        let cond_proj = if let Some(dim) = cond_proj_dim {
            Some(candle_nn::linear_no_bias(dim, in_channels, vb.pp("cond_proj"))?)
        } else {
            None
        };

        let time_embed_dim_out = out_dim.unwrap_or(time_embed_dim);
        let linear_2 = candle_nn::linear(time_embed_dim, time_embed_dim_out, vb.pp("linear_2"))?;

        Ok(Self {
            linear_1,
            linear_2,
            cond_proj,
            act_fn: "silu".to_string(), // Fixed to SiLU as per python default
        })
    }

    /// Forward pass for `TimestepEmbedding`
    pub fn forward(&self, sample: &Tensor, condition: Option<&Tensor>) -> Result<Tensor> {
        let mut sample = sample.clone();

        if let (Some(cond), Some(proj)) = (condition, &self.cond_proj) {
            sample = sample.broadcast_add(&proj.forward(cond)?)?;
        }

        sample = self.linear_1.forward(&sample)?;

        // SiLU activation: x * sigmoid(x)
        sample = sample.broadcast_mul(&candle_nn::ops::sigmoid(&sample)?)?;

        self.linear_2.forward(&sample)
    }
}

/// Upsample1D module
#[derive(Debug, Clone)]
pub struct Upsample1D {
    /// Number of channels
    pub channels: usize,
    /// Number of output channels
    pub out_channels: usize,
    /// Whether to use convolution transpose
    pub use_conv_transpose: bool,
    /// Whether to use convolution
    pub use_conv: bool,
    /// Optional convolution layer
    pub conv: Option<candle_nn::ConvTranspose1d>,
    /// Optional standard 1D conv
    pub conv1d: Option<Conv1d>,
}

impl Upsample1D {
    /// Creates a new `Upsample1D` layer
    pub fn new(
        channels: usize,
        use_conv: bool,
        use_conv_transpose: bool,
        out_channels: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let out_channels = out_channels.unwrap_or(channels);

        let mut conv_transpose = None;
        let mut conv1d = None;

        if use_conv_transpose {
            let config = candle_nn::ConvTranspose1dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            };
            conv_transpose = Some(candle_nn::conv_transpose1d(
                channels,
                out_channels,
                4,
                config,
                vb.pp("conv"),
            )?);
        } else if use_conv {
            let config = candle_nn::Conv1dConfig {
                padding: 1,
                ..Default::default()
            };
            conv1d = Some(candle_nn::conv1d(channels, out_channels, 3, config, vb.pp("conv"))?);
        }

        Ok(Self {
            channels,
            out_channels,
            use_conv_transpose,
            use_conv,
            conv: conv_transpose,
            conv1d,
        })
    }

    /// Forward pass for `Upsample1D`
    pub fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        if inputs.dim(1)? != self.channels {
            candle_core::bail!("Expected inputs to have {} channels", self.channels);
        }

        if self.use_conv_transpose {
            if let Some(conv) = &self.conv {
                return conv.forward(inputs);
            }
        }

        // 1D Nearest neighbor interpolation via broadcasting and reshape
        let dims = inputs.dims();
        let b = dims[0];
        let c = dims[1];
        let l = dims[2];

        // Add a new dimension for the scale factor (2)
        let inputs_unsqueezed = inputs.unsqueeze(3)?; // shape: (b, c, l, 1)
        let broadcasted = inputs_unsqueezed.broadcast_as((b, c, l, 2))?;

        // Contiguous is critical before reshape to flatten the last two dims
        let outputs = broadcasted.contiguous()?.reshape((b, c, l * 2))?;

        if self.use_conv {
            if let Some(conv) = &self.conv1d {
                return conv.forward(&outputs);
            }
        }

        Ok(outputs)
    }
}

/// Stub for BasicTransformerBlock to allow compilation.
/// This will be replaced/expanded when translating transformer.py
#[derive(Debug, Clone)]
pub struct BasicTransformerBlock {
    // Empty stub
}

impl BasicTransformerBlock {
    /// Creates a new stub `BasicTransformerBlock`
    pub fn new(
        _dim: usize,
        _num_attention_heads: usize,
        _attention_head_dim: usize,
        _dropout: f64,
        _activation_fn: &str,
        _vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {})
    }

    /// Forward pass stub
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        _attention_mask: &Tensor,
        _timestep: &Tensor,
    ) -> Result<Tensor> {
        Ok(hidden_states.clone())
    }
}

/// A block representing a stage in the downsampling path
#[derive(Debug, Clone)]
pub struct DownBlock {
    /// ResNet block
    pub resnet: ResnetBlock1D,
    /// Transformer blocks
    pub transformer_blocks: Vec<BasicTransformerBlock>,
    /// Downsample block (convolution)
    pub downsample: Conv1d,
}

/// A block representing a stage in the middle path
#[derive(Debug, Clone)]
pub struct MidBlock {
    /// ResNet block
    pub resnet: ResnetBlock1D,
    /// Transformer blocks
    pub transformer_blocks: Vec<BasicTransformerBlock>,
}

/// A block representing a stage in the upsampling path
#[derive(Debug, Clone)]
pub struct UpBlock {
    /// ResNet block
    pub resnet: ResnetBlock1D,
    /// Transformer blocks
    pub transformer_blocks: Vec<BasicTransformerBlock>,
    /// Upsample block (convolution or ConvTranspose)
    pub upsample: Conv1d,
}

// Since python allows heterogenous module lists (resnet, list(transformers), downsample),
// we use dedicated structs to hold these in Rust for type safety.
// However, upsample can be Upsample1D or Conv1d.
// In the python code:
// upsample = Upsample1D(output_channel, use_conv_transpose=True) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
// Let's create an enum for Downsample and Upsample to handle the variations.

/// Downsample variation enum
#[derive(Debug, Clone)]
pub enum DownsampleVariation {
    /// Custom downsample block
    Downsample1D(Downsample1D),
    /// Standard 1D convolution
    Conv1d(Conv1d),
}

impl DownsampleVariation {
    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Downsample1D(d) => d.forward(x),
            Self::Conv1d(c) => c.forward(x),
        }
    }
}

/// Upsample variation enum
#[derive(Debug, Clone)]
pub enum UpsampleVariation {
    /// Custom upsample block
    Upsample1D(Upsample1D),
    /// Standard 1D convolution
    Conv1d(Conv1d),
}

impl UpsampleVariation {
    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Upsample1D(u) => u.forward(x),
            Self::Conv1d(c) => c.forward(x),
        }
    }
}

/// A typed DownBlock
#[derive(Debug, Clone)]
pub struct TypedDownBlock {
    /// ResNet block
    pub resnet: ResnetBlock1D,
    /// Transformer blocks
    pub transformer_blocks: Vec<BasicTransformerBlock>,
    /// Downsample variation
    pub downsample: DownsampleVariation,
}

/// A typed UpBlock
#[derive(Debug, Clone)]
pub struct TypedUpBlock {
    /// ResNet block
    pub resnet: ResnetBlock1D,
    /// Transformer blocks
    pub transformer_blocks: Vec<BasicTransformerBlock>,
    /// Upsample variation
    pub upsample: UpsampleVariation,
}

/// UNet1DConditional Decoder Model
#[derive(Debug, Clone)]
pub struct Decoder {
    /// Input channels
    pub in_channels: usize,
    /// Output channels
    pub out_channels: usize,

    /// Positional embedding for time
    pub time_embeddings: SinusoidalPosEmb,
    /// MLP for time embedding
    pub time_mlp: TimestepEmbedding,

    /// Down blocks
    pub down_blocks: Vec<TypedDownBlock>,
    /// Mid blocks
    pub mid_blocks: Vec<MidBlock>,
    /// Up blocks
    pub up_blocks: Vec<TypedUpBlock>,

    /// Final 1D block
    pub final_block: Block1D,
    /// Final projection convolution
    pub final_proj: Conv1d,
}

impl Decoder {
    /// Creates a new `Decoder` model
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
        if channels.is_empty() {
            candle_core::bail!("Channels array cannot be empty");
        }

        let time_embeddings = SinusoidalPosEmb::new(in_channels)?;
        let time_embed_dim = channels[0] * 4;

        let time_mlp = TimestepEmbedding::new(
            in_channels,
            time_embed_dim,
            None,
            None,
            vb.pp("time_mlp"),
        )?;

        let mut down_blocks = Vec::new();
        let mut output_channel = in_channels;

        // Down Blocks
        for (i, &c) in channels.iter().enumerate() {
            let input_channel = output_channel;
            output_channel = c;
            let is_last = i == channels.len() - 1;

            let resnet = ResnetBlock1D::new(
                input_channel,
                output_channel,
                time_embed_dim,
                8, // default groups
                vb.pp(format!("down_blocks.{}.0", i)),
            )?;

            let mut transformer_blocks = Vec::new();
            for j in 0..n_blocks {
                let tb = Self::get_block(
                    down_block_type,
                    output_channel,
                    attention_head_dim,
                    num_heads,
                    dropout,
                    act_fn,
                    vb.pp(format!("down_blocks.{}.1.{}", i, j)),
                )?;
                transformer_blocks.push(tb);
            }

            let downsample = if !is_last {
                DownsampleVariation::Downsample1D(Downsample1D::new(output_channel, vb.pp(format!("down_blocks.{}.2", i)))?)
            } else {
                let config = candle_nn::Conv1dConfig {
                    padding: 1,
                    ..Default::default()
                };
                DownsampleVariation::Conv1d(candle_nn::conv1d(output_channel, output_channel, 3, config, vb.pp(format!("down_blocks.{}.2", i)))?)
            };

            down_blocks.push(TypedDownBlock {
                resnet,
                transformer_blocks,
                downsample,
            });
        }

        let mut mid_blocks = Vec::new();
        // Mid Blocks
        let input_channel = *channels.last().unwrap();
        let output_channel = input_channel;

        for i in 0..num_mid_blocks {
            let resnet = ResnetBlock1D::new(
                input_channel,
                output_channel,
                time_embed_dim,
                8,
                vb.pp(format!("mid_blocks.{}.0", i)),
            )?;

            let mut transformer_blocks = Vec::new();
            for j in 0..n_blocks {
                let tb = Self::get_block(
                    mid_block_type,
                    output_channel,
                    attention_head_dim,
                    num_heads,
                    dropout,
                    act_fn,
                    vb.pp(format!("mid_blocks.{}.1.{}", i, j)),
                )?;
                transformer_blocks.push(tb);
            }

            mid_blocks.push(MidBlock { resnet, transformer_blocks });
        }

        let mut up_blocks = Vec::new();
        let mut rev_channels = channels.to_vec();
        rev_channels.reverse();
        rev_channels.push(channels[0]);

        // Up Blocks
        for i in 0..(rev_channels.len() - 1) {
            let input_channel = rev_channels[i];
            let output_channel = rev_channels[i + 1];
            let is_last = i == rev_channels.len() - 2;

            let resnet = ResnetBlock1D::new(
                2 * input_channel,
                output_channel,
                time_embed_dim,
                8,
                vb.pp(format!("up_blocks.{}.0", i)),
            )?;

            let mut transformer_blocks = Vec::new();
            for j in 0..n_blocks {
                let tb = Self::get_block(
                    up_block_type,
                    output_channel,
                    attention_head_dim,
                    num_heads,
                    dropout,
                    act_fn,
                    vb.pp(format!("up_blocks.{}.1.{}", i, j)),
                )?;
                transformer_blocks.push(tb);
            }

            let upsample = if !is_last {
                UpsampleVariation::Upsample1D(Upsample1D::new(output_channel, false, true, None, vb.pp(format!("up_blocks.{}.2", i)))?)
            } else {
                let config = candle_nn::Conv1dConfig {
                    padding: 1,
                    ..Default::default()
                };
                UpsampleVariation::Conv1d(candle_nn::conv1d(output_channel, output_channel, 3, config, vb.pp(format!("up_blocks.{}.2", i)))?)
            };

            up_blocks.push(TypedUpBlock {
                resnet,
                transformer_blocks,
                upsample,
            });
        }

        let final_channel = *rev_channels.last().unwrap();
        let final_block = Block1D::new(final_channel, final_channel, 8, vb.pp("final_block"))?;
        let config = candle_nn::Conv1dConfig {
            padding: 0,
            ..Default::default()
        };
        let final_proj = candle_nn::conv1d(final_channel, out_channels, 1, config, vb.pp("final_proj"))?;

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
    ) -> Result<BasicTransformerBlock> {
        if block_type == "transformer" {
            BasicTransformerBlock::new(
                dim,
                num_heads,
                attention_head_dim,
                dropout,
                act_fn,
                vb,
            )
        } else {
            candle_core::bail!("Unknown block type {} (only transformer supported in stub)", block_type);
        }
    }

    /// Forward pass of the Decoder
    pub fn forward(&self, x: &Tensor, mask: &Tensor, mu: &Tensor, t: &Tensor, spks: Option<&Tensor>, _cond: Option<&Tensor>) -> Result<Tensor> {
        let t_emb = self.time_embeddings.forward(t, 1000.0)?;
        let t_emb = self.time_mlp.forward(&t_emb, None)?;

        // x = pack([x, mu], "b * t")[0] -> concatenate along channel dim (dim=1)
        let mut x = Tensor::cat(&[x, mu], 1)?;

        if let Some(spk) = spks {
            // repeat spks along time dim and concatenate
            let seq_len = x.dim(2)?;
            let spk_repeated = spk.unsqueeze(2)?.broadcast_as((spk.dim(0)?, spk.dim(1)?, seq_len))?;
            x = Tensor::cat(&[&x, &spk_repeated], 1)?;
        }

        let mut hiddens = Vec::new();
        let mut masks = vec![mask.clone()];

        // Down blocks
        for down_block in &self.down_blocks {
            let mask_down = masks.last().unwrap();
            x = down_block.resnet.forward(&x, mask_down, &t_emb)?;

            // rearrange b c t -> b t c
            x = x.transpose(1, 2)?.contiguous()?;
            let mask_down_t = mask_down.squeeze(1)?; // b 1 t -> b t

            for tb in &down_block.transformer_blocks {
                x = tb.forward(&x, &mask_down_t, &t_emb)?;
            }

            // rearrange b t c -> b c t
            x = x.transpose(1, 2)?.contiguous()?;

            hiddens.push(x.clone());

            x = down_block.downsample.forward(&x.broadcast_mul(mask_down)?)?;

            // masks.append(mask_down[:, :, ::2]) -> sub-sample mask by 2
            // Not directly supported with a stride slice in candle natively easily without narrow looping or conv,
            // but since it's a mask, we can just use max pooling or simple interpolation
            // For now, we stub this out or do an explicit sub-sample.
            // Actually, we can use slice to step by 2: mask_down.narrow(2, 0, mask_down.dim(2)? / 2) isn't right.
            // Using a simple reshape hack or conv is possible. Let's use a narrow loop for exactness if small,
            // or just take the first half as a dummy since exact indexing by step is complex.
            // A better way: reshape (b, 1, t/2, 2) and take narrow(3, 0, 1).squeeze(3)
            let t_dim = mask_down.dim(2)?;
            let mask_down_sub = mask_down.reshape((mask_down.dim(0)?, 1, t_dim / 2, 2))?
                .narrow(3, 0, 1)?
                .squeeze(3)?;

            masks.push(mask_down_sub);
        }

        // Pop last mask for mid
        masks.pop();
        let mask_mid = masks.last().unwrap();

        // Mid blocks
        for mid_block in &self.mid_blocks {
            x = mid_block.resnet.forward(&x, mask_mid, &t_emb)?;

            x = x.transpose(1, 2)?.contiguous()?;
            let mask_mid_t = mask_mid.squeeze(1)?;

            for tb in &mid_block.transformer_blocks {
                x = tb.forward(&x, &mask_mid_t, &t_emb)?;
            }

            x = x.transpose(1, 2)?.contiguous()?;
        }

        // Up blocks
        for up_block in &self.up_blocks {
            let mask_up = masks.pop().unwrap();
            let hidden = hiddens.pop().unwrap();

            x = Tensor::cat(&[&x, &hidden], 1)?;
            x = up_block.resnet.forward(&x, &mask_up, &t_emb)?;

            x = x.transpose(1, 2)?.contiguous()?;
            let mask_up_t = mask_up.squeeze(1)?;

            for tb in &up_block.transformer_blocks {
                x = tb.forward(&x, &mask_up_t, &t_emb)?;
            }

            x = x.transpose(1, 2)?.contiguous()?;
            x = up_block.upsample.forward(&x.broadcast_mul(&mask_up)?)?;
        }

        let mask_up = masks.pop().unwrap(); // should be the original mask
        x = self.final_block.forward(&x, &mask_up)?;
        let output = self.final_proj.forward(&x.broadcast_mul(&mask_up)?)?;

        output.broadcast_mul(&mask_up)
    }
}
