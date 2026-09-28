use candle_core::{Device, Tensor};
use std::path::Path;

use crate::s3gen::S3Gen;
use crate::t3::T3;
use crate::tokenizer::{punc_norm, EnTokenizer};

pub struct ChatterboxTTS {
    t3_model: T3,
    s3gen_model: S3Gen,
    tokenizer: EnTokenizer,
    device: Device,
    pub sr: u32,
}

impl ChatterboxTTS {
    pub fn new(t3_model: T3, s3gen_model: S3Gen, tokenizer: EnTokenizer, device: Device) -> Self {
        Self {
            t3_model,
            s3gen_model,
            tokenizer,
            device,
            sr: 24000, // S3Gen sample rate
        }
    }

    pub fn generate(&self, text: &str, audio_prompt_path: Option<&str>) -> anyhow::Result<Tensor> {
        let normalized_text = punc_norm(text);
        let text_tokens = self
            .tokenizer
            .text_to_tokens(&normalized_text, &self.device)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let audio_cond = if let Some(_path) = audio_prompt_path {
            // Placeholder: load and process audio prompt to mel-spectrogram
            // For now just passing None as the dummy doesn't use it
            None
        } else {
            None
        };

        let speech_tokens = self
            .t3_model
            .generate(&text_tokens, audio_cond.as_ref())
            .map_err(|e| anyhow::anyhow!("T3 generation failed: {}", e))?;

        let audio_waveform = self
            .s3gen_model
            .generate(&speech_tokens)
            .map_err(|e| anyhow::anyhow!("S3Gen generation failed: {}", e))?;

        Ok(audio_waveform)
    }

    pub fn save_wav<P: AsRef<Path>>(&self, path: P, waveform: &Tensor) -> anyhow::Result<()> {
        let waveform = waveform.squeeze(0)?.squeeze(0)?;
        let waveform_vec = waveform
            .to_vec1::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to convert tensor to vec: {}", e))?;

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sr,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = hound::WavWriter::create(path, spec)?;
        for sample in waveform_vec {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;

        Ok(())
    }
}
