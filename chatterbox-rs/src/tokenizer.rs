use candle_core::{Device, Result as CandleResult, Tensor};
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

pub const SOT: &str = "[START]";
pub const EOT: &str = "[STOP]";
pub const UNK: &str = "[UNK]";
pub const SPACE: &str = "[SPACE]";

pub struct EnTokenizer {
    tokenizer: HfTokenizer,
}

impl EnTokenizer {
    pub fn from_file<P: AsRef<Path>>(vocab_file_path: P) -> anyhow::Result<Self> {
        let tokenizer = HfTokenizer::from_file(vocab_file_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let vocab = tokenizer.get_vocab(true);
        if !vocab.contains_key(SOT) || !vocab.contains_key(EOT) {
            anyhow::bail!("Vocabulary must contain [START] and [STOP] tokens");
        }

        Ok(Self { tokenizer })
    }

    pub fn text_to_tokens(&self, text: &str, device: &Device) -> CandleResult<Tensor> {
        let text_tokens = self
            .encode(text)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let text_tokens_u32: Vec<u32> = text_tokens.into_iter().map(|t| t).collect();
        let tensor = Tensor::new(text_tokens_u32, device)?;
        tensor.unsqueeze(0)
    }

    pub fn encode(&self, txt: &str) -> anyhow::Result<Vec<u32>> {
        let txt = txt.replace(' ', SPACE);
        let encoding = self
            .tokenizer
            .encode(txt, true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, seq: &[u32]) -> anyhow::Result<String> {
        let txt = self
            .tokenizer
            .decode(seq, false)
            .map_err(|e| anyhow::anyhow!("Failed to decode: {}", e))?;
        let txt = txt.replace(' ', "");
        let txt = txt.replace(SPACE, " ");
        Ok(txt)
    }
}

pub fn punc_norm(text: &str) -> String {
    if text.is_empty() {
        return "You need to add some text for me to talk.".to_string();
    }

    let mut text = text.to_string();

    // Capitalize first character
    if let Some(first) = text.chars().next() {
        if first.is_lowercase() {
            let mut c = text.chars();
            text = match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            };
        }
    }

    // Remove multiple space chars
    let words: Vec<&str> = text.split_whitespace().collect();
    text = words.join(" ");

    let punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ];

    for (old, new) in punc_to_replace {
        text = text.replace(old, new);
    }

    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_punc_norm() {
        assert_eq!(punc_norm(""), "You need to add some text for me to talk.");
        assert_eq!(punc_norm("hello world..."), "Hello world, ");
        assert_eq!(punc_norm("this   has  many spaces"), "This has many spaces");
        assert_eq!(punc_norm("test:"), "Test,");
    }
}
