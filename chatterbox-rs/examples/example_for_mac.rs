use candle_core::Device;
use chatterbox_rs::config::T3Config;
use chatterbox_rs::s3gen::S3Gen;
use chatterbox_rs::t3::T3;
use chatterbox_rs::tokenizer::EnTokenizer;
use chatterbox_rs::tts::ChatterboxTTS;

fn main() -> anyhow::Result<()> {
    // Detect device (Mac with M1/M2/M3/M4)
    // candle-core features for metal can be checked or just initialized
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);

    println!("Using device: {:?}", device);

    // Initialize mock models for demonstration purposes,
    let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
    let t3_config = T3Config::english_only();

    let t3 = T3::load(vb.clone(), t3_config).map_err(|e| anyhow::anyhow!("T3 Error: {}", e))?;
    let s3gen = S3Gen::load(vb.clone()).map_err(|e| anyhow::anyhow!("S3Gen Error: {}", e))?;
    let tokenizer = EnTokenizer::new("tokenizer.json").unwrap_or_else(|_| {
        println!("Warning: tokenizer.json not found, using empty mock tokenizer");
        EnTokenizer::mock()
    });

    let model = ChatterboxTTS::new(t3, s3gen, tokenizer, device.clone());

    let text = "Today is the day. I want to move like a titan at dawn, sweat like a god forging lightning. No more excuses. From now on, my mornings will be temples of discipline. I am going to work out like the gods… every damn day.";

    // If you want to synthesize with a different voice, specify the audio prompt
    let audio_prompt_path = "YOUR_FILE.wav";

    // Using default mock implementation of generate
    let wav = model.generate(text, Some(audio_prompt_path))?;

    model.save_wav("test-2.wav", &wav)?;

    println!("Synthesis completed and saved to test-2.wav");

    Ok(())
}
