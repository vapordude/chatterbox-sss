#[derive(Debug, Clone, PartialEq)]
pub struct VoiceEncConfig {
    pub num_mels: usize,
    pub sample_rate: usize,
    pub speaker_embed_size: usize,
    pub ve_hidden_size: usize,
    pub flatten_lstm_params: bool,
    pub n_fft: usize,
    pub hop_size: usize,
    pub win_size: usize,
    pub fmax: f64,
    pub fmin: f64,
    pub preemphasis: f64,
    pub mel_power: f64,
    pub mel_type: String,
    pub normalized_mels: bool,
    pub ve_partial_frames: usize,
    pub ve_final_relu: bool,
    pub stft_magnitude_min: f64,
}

impl Default for VoiceEncConfig {
    fn default() -> Self {
        Self {
            num_mels: 40,
            sample_rate: 16000,
            speaker_embed_size: 256,
            ve_hidden_size: 256,
            flatten_lstm_params: false,
            n_fft: 400,
            hop_size: 160,
            win_size: 400,
            fmax: 8000.0,
            fmin: 0.0,
            preemphasis: 0.0,
            mel_power: 2.0,
            mel_type: "amp".to_string(),
            normalized_mels: false,
            ve_partial_frames: 160,
            ve_final_relu: true,
            stft_magnitude_min: 1e-4,
        }
    }
}
