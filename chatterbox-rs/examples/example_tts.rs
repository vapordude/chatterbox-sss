use candle_core::Device;
use chatterbox_rs::config::T3Config;
use chatterbox_rs::s3gen::S3Gen;
use chatterbox_rs::t3::T3;
use chatterbox_rs::tokenizer::EnTokenizer;
use chatterbox_rs::tts::ChatterboxTTS;

fn main() -> anyhow::Result<()> {
    // Automatically detect the best available device
    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else if candle_core::utils::metal_is_available() {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };

    println!("Using device: {:?}", device);

    // Initialize mock models for demonstration purposes
    let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
    let t3_config = T3Config::english_only();

    let t3 = T3::load(vb.clone(), t3_config).map_err(|e| anyhow::anyhow!("T3 Error: {}", e))?;
    let s3gen = S3Gen::load(vb.clone()).map_err(|e| anyhow::anyhow!("S3Gen Error: {}", e))?;
    let tokenizer = EnTokenizer::new("tokenizer.json").unwrap_or_else(|_| {
        println!("Warning: tokenizer.json not found, using empty mock tokenizer");
        EnTokenizer::mock()
    });

    let model = ChatterboxTTS::new(t3, s3gen, tokenizer, device.clone());

    let text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill.";

    // Generate without audio prompt
    let wav1 = model.generate(text, None)?;
    model.save_wav("test-1.wav", &wav1)?;
    println!("Synthesis completed and saved to test-1.wav");

    // We don't have Multilingual TTS mocked out yet, skipping that part from the python example.

    let audio_prompt_path = "YOUR_FILE.wav";
    // Usually we would check if path exists, but our mock `generate` ignores it anyway.
    let wav3 = model.generate(text, Some(audio_prompt_path))?;
    model.save_wav("test-3.wav", &wav3)?;
    println!("Synthesis completed and saved to test-3.wav");

    Ok(())
}
