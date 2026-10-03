use candle_core::{Result, Tensor};
use candle_nn::{conv1d, linear, Conv1d, Conv1dConfig, Linear, Module, VarBuilder};

// Helper struct for weight_norm wrapping a Conv1d
// To implement true weight_norm requires maintaining v and g params,
// for inference we just need the regular Conv1d if parameters are already fused,
// or we load them normally. We will just wrap Conv1d here for inference.
pub struct WeightNormConv1d {
    conv: Conv1d,
}

impl WeightNormConv1d {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding,
            ..Default::default()
        };
        // For PyTorch's weight_norm, there are 'weight_v' and 'weight_g'.
        // Typically for inference they might be fused into 'weight'.
        // Assuming weights are fused or can be handled as a standard conv.
        let conv = conv1d(in_channels, out_channels, kernel_size, cfg, vb)?;
        Ok(Self { conv })
    }
}

impl Module for WeightNormConv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

pub struct CondNet {
    layers: Vec<WeightNormConv1d>,
}

impl CondNet {
    pub fn load(vb: VarBuilder, in_channels: usize, cond_channels: usize) -> Result<Self> {
        let mut layers = Vec::new();
        // The original has 5 Conv1d layers wrapped in weight_norm.
        // Sequential indices: 0, 2, 4, 6, 8
        layers.push(WeightNormConv1d::load(
            vb.pp("0"),
            in_channels,
            cond_channels,
            3,
            1,
        )?);
        layers.push(WeightNormConv1d::load(
            vb.pp("2"),
            cond_channels,
            cond_channels,
            3,
            1,
        )?);
        layers.push(WeightNormConv1d::load(
            vb.pp("4"),
            cond_channels,
            cond_channels,
            3,
            1,
        )?);
        layers.push(WeightNormConv1d::load(
            vb.pp("6"),
            cond_channels,
            cond_channels,
            3,
            1,
        )?);
        layers.push(WeightNormConv1d::load(
            vb.pp("8"),
            cond_channels,
            cond_channels,
            3,
            1,
        )?);

        Ok(Self { layers })
    }
}

impl Module for CondNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out)?;
            out = out.elu(1.0)?; // nn.ELU()
        }
        Ok(out)
    }
}

pub struct ConvRNNF0Predictor {
    condnet: CondNet,
    classifier: Linear,
    pub num_class: usize,
}

impl ConvRNNF0Predictor {
    pub fn load(
        vb: VarBuilder,
        num_class: usize,
        in_channels: usize,
        cond_channels: usize,
    ) -> Result<Self> {
        let condnet = CondNet::load(vb.pp("condnet"), in_channels, cond_channels)?;
        let classifier = linear(cond_channels, num_class, vb.pp("classifier"))?;

        Ok(Self {
            condnet,
            classifier,
            num_class,
        })
    }
}

impl Module for ConvRNNF0Predictor {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.condnet.forward(x)?;
        let x = x.transpose(1, 2)?;
        let x = self.classifier.forward(&x)?;
        x.squeeze(2)?.abs() // squeeze(-1) and torch.abs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_f0_predictor() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let model = ConvRNNF0Predictor::load(vb, 1, 80, 512).unwrap();
        let input = Tensor::randn(0.0f32, 1.0f32, (2, 80, 100), &device).unwrap();

        let out = model.forward(&input).unwrap();
        assert_eq!(out.dims(), &[2, 100]);
        println!("test_f0_predictor pass");
    }
}
