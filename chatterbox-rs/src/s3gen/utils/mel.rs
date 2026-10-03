use candle_core::{Device, Result, Tensor};

// Dynamic range compression
pub fn dynamic_range_compression(x: &Tensor, c: f64, clip_val: f64) -> Result<Tensor> {
    let clamped = x.maximum(clip_val)?;
    let multiplied = (clamped * c)?;
    multiplied.log()
}

// Spectral normalize
pub fn spectral_normalize(magnitudes: &Tensor) -> Result<Tensor> {
    dynamic_range_compression(magnitudes, 1.0, 1e-5)
}

fn hz_to_mel(frequencies: &Tensor, htk: bool) -> Result<Tensor> {
    if htk {
        // 2595.0 * np.log10(1.0 + frequencies / 700.0)
        let f_scaled = (frequencies / 700.0)?;
        let f_plus_one = (f_scaled + 1.0)?;
        let log10_f = (f_plus_one.log()? / 10f64.ln())?;
        return log10_f * 2595.0;
    }

    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;

    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = 6.4f64.ln() / 27.0;

    let mels = ((frequencies - f_min)? / f_sp)?;

    // Handle linear and log scale switching
    let device = frequencies.device();
    // Assuming frequencies is 1D or 0D. In rust we will just use map
    let frequencies_f64 = frequencies.to_vec1::<f64>()?;
    let mut mels_f64 = mels.to_vec1::<f64>()?;

    for i in 0..frequencies_f64.len() {
        if frequencies_f64[i] >= min_log_hz {
            mels_f64[i] = min_log_mel + (frequencies_f64[i] / min_log_hz).ln() / logstep;
        }
    }

    Tensor::new(mels_f64, device)
}

fn mel_to_hz(mels: &Tensor, htk: bool) -> Result<Tensor> {
    if htk {
        // 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        let device = mels.device();
        let mels_f64 = mels.to_vec1::<f64>()?;
        let mut freqs_f64 = vec![0.0; mels_f64.len()];
        for i in 0..mels_f64.len() {
            freqs_f64[i] = 700.0 * (10f64.powf(mels_f64[i] / 2595.0) - 1.0);
        }
        return Tensor::new(freqs_f64, device);
    }

    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let freqs = ((mels * f_sp)? + f_min)?;

    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = 6.4f64.ln() / 27.0;

    let device = mels.device();
    let mels_f64 = mels.to_vec1::<f64>()?;
    let mut freqs_f64 = freqs.to_vec1::<f64>()?;

    for i in 0..mels_f64.len() {
        if mels_f64[i] >= min_log_mel {
            freqs_f64[i] = min_log_hz * (logstep * (mels_f64[i] - min_log_mel)).exp();
        }
    }

    Tensor::new(freqs_f64, device)
}

fn mel_frequencies(
    n_mels: usize,
    fmin: f64,
    fmax: f64,
    htk: bool,
    device: &Device,
) -> Result<Tensor> {
    let min_mel = hz_to_mel(&Tensor::new(vec![fmin], device)?, htk)?.to_vec1::<f64>()?[0];
    let max_mel = hz_to_mel(&Tensor::new(vec![fmax], device)?, htk)?.to_vec1::<f64>()?[0];

    // np.linspace(min_mel, max_mel, n_mels)
    let step = if n_mels > 1 {
        (max_mel - min_mel) / ((n_mels - 1) as f64)
    } else {
        0.0
    };
    let mut mels = Vec::with_capacity(n_mels);
    for i in 0..n_mels {
        mels.push(min_mel + step * (i as f64));
    }

    mel_to_hz(&Tensor::new(mels, device)?, htk)
}

pub fn mel(
    sr: f64,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
    htk: bool,
    norm: bool,
    device: &Device,
) -> Result<Tensor> {
    let fftfreqs = {
        let step = (sr / 2.0) / ((n_fft / 2) as f64);
        let mut freqs = Vec::with_capacity(1 + n_fft / 2);
        for i in 0..=n_fft / 2 {
            freqs.push(step * (i as f64));
        }
        freqs
    };

    let mel_f = mel_frequencies(n_mels + 2, fmin, fmax, htk, device)?.to_vec1::<f64>()?;

    let mut fdiff = Vec::with_capacity(mel_f.len() - 1);
    for i in 0..mel_f.len() - 1 {
        fdiff.push(mel_f[i + 1] - mel_f[i]);
    }

    let mut weights = vec![vec![0.0f32; 1 + n_fft / 2]; n_mels];

    for i in 0..n_mels {
        for j in 0..=n_fft / 2 {
            let ramp = mel_f[i] - fftfreqs[j];
            let ramp_next_2 = mel_f[i + 2] - fftfreqs[j];

            let lower = -ramp / fdiff[i];
            let upper = ramp_next_2 / fdiff[i + 1];

            let val = lower.min(upper).max(0.0);
            weights[i][j] = val as f32;
        }
    }

    if norm {
        for i in 0..n_mels {
            let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
            for j in 0..=n_fft / 2 {
                weights[i][j] *= enorm as f32;
            }
        }
    }

    let mut flat_weights = Vec::with_capacity(n_mels * (1 + n_fft / 2));
    for i in 0..n_mels {
        flat_weights.extend_from_slice(&weights[i]);
    }

    Tensor::from_vec(flat_weights, (n_mels, 1 + n_fft / 2), device)
}

#[cfg(test)]
mod tests {
    use super::super::mel;
    use candle_core::Device;

    #[test]
    fn test_mel() {
        let device = Device::Cpu;
        let weights = mel::mel(24000.0, 1920, 80, 0.0, 8000.0, false, true, &device).unwrap();
        assert_eq!(weights.dims(), &[80, 961]);
        println!("test_mel pass");
    }
}
