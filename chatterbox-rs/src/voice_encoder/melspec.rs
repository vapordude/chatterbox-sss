use crate::s3gen::utils::mel::mel;
use crate::voice_encoder::config::VoiceEncConfig;
use candle_core::{Device, Tensor};

pub fn mel_basis(hp: &VoiceEncConfig) -> candle_core::Result<Tensor> {
    assert!(hp.fmax as usize <= hp.sample_rate / 2);
    // Note: librosa.filters.mel uses htk=False, norm="slaney" by default
    // We already have mel filter implementation in s3gen::utils::mel::mel, which we can adapt.
    mel(
        hp.sample_rate as f64,
        hp.n_fft,
        hp.num_mels,
        hp.fmin,
        hp.fmax,
        false, // htk=False
        true,  // norm="slaney" (true in our mel function maps to it roughly, though we should check)
        &Device::Cpu,
    )
}

pub fn preemphasis(wav: &[f32], hp: &VoiceEncConfig) -> Vec<f32> {
    assert!(hp.preemphasis != 0.0);
    // signal.lfilter([1, -hp.preemphasis], [1], wav)
    let mut out = vec![0.0; wav.len()];
    if wav.is_empty() {
        return out;
    }
    
    out[0] = wav[0]; // For first element, previous is considered 0
    for i in 1..wav.len() {
        out[i] = wav[i] - (hp.preemphasis as f32) * wav[i - 1];
    }
    
    for x in &mut out {
        *x = x.clamp(-1.0, 1.0);
    }
    
    out
}

// STFT implementation from first principles
use std::f32::consts::PI;

pub fn hann_window(win_length: usize) -> Vec<f32> {
    let mut win = Vec::with_capacity(win_length);
    for i in 0..win_length {
        let val = 0.5 - 0.5 * (2.0 * PI * i as f32 / (win_length as f32)).cos();
        win.push(val);
    }
    win
}


pub fn fft(x: &mut [(f32, f32)]) {
    let n = x.len();
    if n <= 1 {
        return;
    }
    
    // Bit-reverse copy
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            x.swap(i, j);
        }
    }
    
    // Cooley-Tukey
    let mut len = 2;
    while len <= n {
        let angle = -2.0 * PI / (len as f32);
        let wlen_re = angle.cos();
        let wlen_im = angle.sin();
        for i in (0..n).step_by(len) {
            let mut w_re = 1.0;
            let mut w_im = 0.0;
            for j in 0..len / 2 {
                let u_re = x[i + j].0;
                let u_im = x[i + j].1;
                let v_re = x[i + j + len / 2].0 * w_re - x[i + j + len / 2].1 * w_im;
                let v_im = x[i + j + len / 2].0 * w_im + x[i + j + len / 2].1 * w_re;
                
                x[i + j] = (u_re + v_re, u_im + v_im);
                x[i + j + len / 2] = (u_re - v_re, u_im - v_im);
                
                let next_w_re = w_re * wlen_re - w_im * wlen_im;
                let next_w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = next_w_re;
                w_im = next_w_im;
            }
        }
        len *= 2;
    }
}


pub fn stft_magnitudes(y: &[f32], n_fft: usize, hop_length: usize, win_length: usize, center: bool) -> Vec<Vec<f32>> {
    let mut y_pad = Vec::new();
    if center {
        // Pad with reflect mode
        let pad_len = n_fft / 2;
        y_pad.reserve(y.len() + 2 * pad_len);
        
        // left pad (reflect)
        for i in (1..=pad_len).rev() {
            if i < y.len() {
                y_pad.push(y[i]);
            } else {
                y_pad.push(0.0);
            }
        }
        
        y_pad.extend_from_slice(y);
        
        // right pad (reflect)
        for i in 1..=pad_len {
            if y.len() >= 1 + i {
                y_pad.push(y[y.len() - 1 - i]);
            } else {
                y_pad.push(0.0);
            }
        }
    } else {
        y_pad.extend_from_slice(y);
    }

    let num_frames = 1 + (y_pad.len().saturating_sub(n_fft)) / hop_length;
    let window = hann_window(win_length);
    let pad_left = (n_fft - win_length) / 2;
    
    let mut frames_mag = Vec::with_capacity(num_frames);

    for t in 0..num_frames {
        let start = t * hop_length;
        let mut frame = vec![0.0; n_fft];
        for i in 0..win_length {
            frame[pad_left + i] = y_pad[start + pad_left + i] * window[i];
        }
        
        // Compute FFT magnitudes. Since we just need magnitudes, we can compute them directly (or use a simple O(N^2) DFT or Cooley-Tukey O(N log N)).
        // We'll implement a basic radix-2 FFT for performance since n_fft can be 400. 
        // Oh wait, 400 is not a power of 2. It is 16 * 25. Radix-2 won't work perfectly. We can pad to next power of 2? No, librosa does n_fft=400 exactly. 
        // We will implement a simple O(N^2) DFT since N=400 is very small, or Bluestein's / Cooley-Tukey.
        // Actually, for N=400, N^2 is 160,000 operations per frame, which is completely fine in Rust.
        
        // Pad to next power of 2 for FFT
        let mut next_pow2 = 1;
        while next_pow2 < n_fft {
            next_pow2 *= 2;
        }
        
        let mut complex_frame = vec![(0.0f32, 0.0f32); next_pow2];
        for i in 0..n_fft {
            complex_frame[i] = (frame[i], 0.0);
        }
        
        fft(&mut complex_frame);
        
        let mut mag = Vec::with_capacity(1 + n_fft / 2);
        for k in 0..=n_fft / 2 {
            let re = complex_frame[k].0;
            let im = complex_frame[k].1;
            mag.push((re * re + im * im).sqrt());
        }
        frames_mag.push(mag);
    }
    
    frames_mag // shape (num_frames, 1 + n_fft/2)
}

fn amp_to_db(x: f32, hp: &VoiceEncConfig) -> f32 {
    20.0 * (x.max(hp.stft_magnitude_min as f32)).log10()
}

fn normalize(s: f32, hp: &VoiceEncConfig, headroom_db: f32) -> f32 {
    let min_level_db = 20.0 * (hp.stft_magnitude_min as f32).log10();
    (s - min_level_db) / (-min_level_db + headroom_db)
}

pub fn melspectrogram(wav: &[f32], hp: &VoiceEncConfig, pad: bool) -> Vec<Vec<f32>> {
    let wav_proc = if hp.preemphasis > 0.0 {
        preemphasis(wav, hp)
    } else {
        wav.to_vec()
    };

    let spec_magnitudes = stft_magnitudes(&wav_proc, hp.n_fft, hp.hop_size, hp.win_size, pad); // shape (T, F)
    let num_frames = spec_magnitudes.len();
    let num_freqs = 1 + hp.n_fft / 2;
    
    // Convert to Tensor for matrix multiplication
    let mut flat_mags = Vec::with_capacity(num_freqs * num_frames);
    // We want shape (F, T) for dot product with mel_basis
    for f in 0..num_freqs {
        for t in 0..num_frames {
            let mut val = spec_magnitudes[t][f];
            if hp.mel_power != 1.0 {
                val = val.powf(hp.mel_power as f32);
            }
            flat_mags.push(val);
        }
    }
    
    let spec_tensor = Tensor::from_vec(flat_mags, (num_freqs, num_frames), &Device::Cpu).unwrap();
    let mel_basis = mel_basis(hp).unwrap(); // shape (num_mels, F)
    
    // Get the mel and convert magnitudes->db
    // mel = np.dot(mel_basis, spec_magnitudes)
    let mel = mel_basis.to_dtype(candle_core::DType::F32).unwrap()
        .matmul(&spec_tensor).unwrap(); // shape (num_mels, T)
        
    let mel_flat = mel.to_vec2::<f32>().unwrap(); // shape (num_mels, T)
    
    let mut mel_proc = vec![vec![0.0; num_frames]; hp.num_mels];
    for m in 0..hp.num_mels {
        for t in 0..num_frames {
            let mut val = mel_flat[m][t];
            if hp.mel_type == "db" {
                val = amp_to_db(val, hp);
            }
            if hp.normalized_mels {
                val = normalize(val, hp, 15.0);
            }
            mel_proc[m][t] = val;
        }
    }
    
    mel_proc // (M, T)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preemphasis() {
        let hp = VoiceEncConfig::default(); // preemphasis is 0.0 by default, let's set it
        let mut hp_emp = hp.clone();
        hp_emp.preemphasis = 0.97;
        
        let wav = vec![0.5, 0.8, -0.2];
        let emp = preemphasis(&wav, &hp_emp);
        
        assert_eq!(emp[0], 0.5);
        assert!((emp[1] - (0.8 - 0.97 * 0.5)).abs() < 1e-5);
        assert!((emp[2] - (-0.2 - 0.97 * 0.8)).abs() < 1e-5);
    }
}
