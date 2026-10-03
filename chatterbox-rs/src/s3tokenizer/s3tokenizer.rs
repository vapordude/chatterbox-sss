use crate::s3gen::utils::mel::mel;
use candle_core::{Device, Result as CandleResult, Tensor};
use std::f32::consts::PI;

pub const S3_SR: usize = 16_000;
pub const S3_HOP: usize = 160;
pub const S3_TOKEN_HOP: usize = 640;
pub const S3_TOKEN_RATE: usize = 25;
pub const SPEECH_VOCAB_SIZE: usize = 6561;

pub struct S3Tokenizer {
    n_fft: usize,
    mel_filters: Tensor,
    window: Tensor,
}

impl S3Tokenizer {
    pub fn new(n_mels: usize, device: &Device) -> CandleResult<Self> {
        let n_fft = 400;

        let mel_filters = mel(
            S3_SR as f64,
            n_fft,
            n_mels,
            0.0,
            (S3_SR as f64) / 2.0,
            false,
            true,
            device,
        )?;

        let mut window_data = Vec::with_capacity(n_fft);
        for i in 0..n_fft {
            let val =
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * (i as f64) / ((n_fft - 1) as f64)).cos();
            window_data.push(val as f32);
        }
        let window = Tensor::from_vec(window_data, n_fft, device)?;

        Ok(Self {
            n_fft,
            mel_filters,
            window,
        })
    }

    pub fn pad(&self, wavs: &[Tensor], sr: f64) -> CandleResult<Vec<Tensor>> {
        let mut processed_wavs = Vec::new();
        for wav in wavs {
            let mut wav = wav.clone();
            if wav.dims().len() == 1 {
                wav = wav.unsqueeze(0)?;
            }

            let seq_len = wav.dims().last().unwrap();
            let mut n_tokens = (*seq_len as f64 / sr) * (S3_TOKEN_RATE as f64);
            n_tokens = n_tokens.ceil();

            let intended_wav_len = (n_tokens * (sr / S3_TOKEN_RATE as f64)) as usize;

            if intended_wav_len > *seq_len {
                let pad_len = intended_wav_len - seq_len;
                let zeros = Tensor::zeros((1, pad_len), wav.dtype(), wav.device())?;
                wav = Tensor::cat(&[&wav, &zeros], 1)?;
            }

            processed_wavs.push(wav);
        }
        Ok(processed_wavs)
    }

    pub fn prepare_audio(&self, wavs: &[Tensor]) -> CandleResult<Vec<Tensor>> {
        let mut processed_wavs = Vec::new();
        for wav in wavs {
            let mut wav = wav.clone();
            if wav.dims().len() == 1 {
                wav = wav.unsqueeze(0)?;
            }
            processed_wavs.push(wav);
        }
        Ok(processed_wavs)
    }

    pub fn log_mel_spectrogram(&self, audio: &Tensor, padding: usize) -> CandleResult<Tensor> {
        let mut audio = audio.clone();
        if padding > 0 {
            let zeros = Tensor::zeros((1, padding), audio.dtype(), audio.device())?;
            audio = Tensor::cat(&[&audio, &zeros], 1)?;
        }

        let _audio_len = audio.dims().last().unwrap();
        let hop_length = S3_HOP;
        let n_fft = self.n_fft;

        // Number of frames: 1 + (audio_len - n_fft) / hop_length if centered=false,
        // wait, torch.stft by default uses center=True, meaning audio is padded with n_fft // 2 on both sides.
        // Let's implement a simple stft with center=True
        let pad_len = n_fft / 2;
        let zeros_pad = Tensor::zeros((1, pad_len), audio.dtype(), audio.device())?;
        audio = Tensor::cat(&[&zeros_pad, &audio, &zeros_pad], 1)?;
        let padded_len = audio.dims().last().unwrap();

        let num_frames = 1 + (padded_len - n_fft) / hop_length;
        let audio_data = audio.squeeze(0)?.to_vec1::<f32>()?;
        let window_data = self.window.to_vec1::<f32>()?;

        // Compute magnitude squared (power spectrum)
        // Only return n_fft / 2 + 1 bins, but torch.stft[..., :-1] drops the highest frequency bin.
        // So we keep n_fft / 2 bins.
        let num_bins = n_fft / 2 + 1;
        let mut magnitudes_sq = Vec::with_capacity(num_bins * num_frames);

        for t in 0..num_frames {
            let start = t * hop_length;
            let _end = start + n_fft;
            let mut frame = vec![0.0; n_fft];
            for i in 0..n_fft {
                frame[i] = audio_data[start + i] * window_data[i];
            }

            // DFT
            for k in 0..num_bins {
                let mut real = 0.0;
                let mut imag = 0.0;
                let angle_factor = -2.0 * PI * (k as f32) / (n_fft as f32);
                for n in 0..n_fft {
                    let angle = angle_factor * (n as f32);
                    real += frame[n] * angle.cos();
                    imag += frame[n] * angle.sin();
                }
                magnitudes_sq.push(real * real + imag * imag);
            }
        }

        let num_frames_minus_1 = num_frames - 1;
        let mags_tensor = Tensor::from_vec(magnitudes_sq, (num_frames, num_bins), audio.device())?;
        let mags_tensor = mags_tensor.narrow(0, 0, num_frames_minus_1)?;
        // Transpose to match (F, T) - F = num_bins, T = num_frames
        let mags_tensor = mags_tensor.transpose(0, 1)?;
        let mags_tensor = mags_tensor.unsqueeze(0)?; // [B=1, F, T]

        // mel_filters @ magnitudes
        // self.mel_filters is (n_mels, 1 + n_fft / 2). Oh, wait. mel.rs returns 1 + n_fft / 2 bins.
        // python does `magnitudes = stft[..., :-1].abs()**2` which means it keeps `n_fft / 2` bins.
        // wait, let me check torch.stft. It returns `1 + n_fft / 2` bins. [..., :-1] removes the last bin.
        // So num_bins = n_fft / 2.
        // Let's slice mel_filters to [n_mels, n_fft / 2]
        let mel_filters_sliced = self.mel_filters.clone(); // The shape is already (n_mels, num_bins)
        let mel_filters_sliced = mel_filters_sliced.unsqueeze(0)?; // [1, n_mels, num_bins]

        // mel_spec = mel_filters @ magnitudes
        let mel_spec = mel_filters_sliced.matmul(&mags_tensor)?;

        // log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        let log_spec = mel_spec.maximum(1e-10f64)?;
        // ln(x) / ln(10) = log10(x)
        let log_spec = (log_spec.log()? / 10f64.ln())?;

        // log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        let max_val = log_spec.max_keepdim(0)?.max_keepdim(1)?.max_keepdim(2)?;
        let threshold = (max_val - 8.0)?;
        let threshold_broadcast = threshold.broadcast_as(log_spec.dims())?;
        let log_spec = log_spec.maximum(&threshold_broadcast)?;

        // log_spec = (log_spec + 4.0) / 4.0
        let log_spec = ((log_spec + 4.0)? / 4.0)?;

        Ok(log_spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_pad() -> CandleResult<()> {
        let device = Device::Cpu;
        let s3 = S3Tokenizer::new(80, &device)?;

        let wav = Tensor::ones((1, 16000), candle_core::DType::F32, &device)?;
        let padded = s3.pad(&[wav.clone()], S3_SR as f64)?;

        assert_eq!(padded.len(), 1);
        assert_eq!(padded[0].dims(), &[1, 16000]);

        let wav2 = Tensor::ones((1, 8000), candle_core::DType::F32, &device)?;
        let padded2 = s3.pad(&[wav2.clone()], S3_SR as f64)?;
        assert_eq!(padded2[0].dims(), &[1, 8320]); // 8000 -> 12.5 tokens -> ceil -> 13 tokens -> 13 * (16000/25) = 13 * 640 = 8320
        Ok(())
    }

    #[test]
    fn test_prepare_audio() -> CandleResult<()> {
        let device = Device::Cpu;
        let s3 = S3Tokenizer::new(80, &device)?;

        let wav = Tensor::ones(16000, candle_core::DType::F32, &device)?;
        let prepared = s3.prepare_audio(&[wav])?;
        assert_eq!(prepared[0].dims(), &[1, 16000]);
        Ok(())
    }

    #[test]
    fn test_log_mel_spectrogram() -> CandleResult<()> {
        let device = Device::Cpu;
        let s3 = S3Tokenizer::new(80, &device)?;

        let wav = Tensor::randn(0f32, 1f32, (1, 16000), &device)?;
        let mel = s3.log_mel_spectrogram(&wav, 0)?;

        // num_frames for 16000 with center pad:
        // padded_len = 16000 + 400 = 16400
        // num_frames = 1 + (16400 - 400) / 160 = 1 + 16000 / 160 = 101
        assert_eq!(mel.dims(), &[1, 80, 100]);
        Ok(())
    }
}
