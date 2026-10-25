//! Usage: `cargo run --release --example demo -- <models dir> [word ...]`
//!
//! Without words, reads `word<TAB>PH PH ...` lines from stdin and prints
//! tab-separated results (used to compare against the Python package).

use std::io::BufRead;
use std::path::PathBuf;

use e2k::{AccentPredictor, NGram, Strategy, C2K, P2K};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().ok_or("usage: demo <models dir> [word ...]")?);
    let words: Vec<String> = args.collect();

    let c2k = C2K::from_path(dir.join("model-c2k.safetensors"))?;
    let p2k = P2K::from_path(dir.join("model-p2k.safetensors"))?;
    let ap = AccentPredictor::from_path(dir.join("accent.safetensors"))?;
    let ngram = NGram::from_path(dir.join("ngram.json.zip"))?;

    if !words.is_empty() {
        for word in &words {
            let katakana = c2k.infer(word);
            println!("{word}");
            println!(
                "  N-gram: {}",
                if ngram.is_valid(word) {
                    "valid"
                } else {
                    "invalid"
                }
            );
            println!("  As is: {}", ngram.as_is(word));
            println!("  C2K: {katakana}, accent: {}", ap.predict(&katakana));
            println!("  C2K (top_k): {}", c2k.infer_with(word, Strategy::top_k()));
            println!(
                "  C2K (top_p): {}",
                c2k.infer_with(word, Strategy::TopP { p: 0.7, t: 2.0 })
            );
        }
        return Ok(());
    }

    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let (word, phonemes) = line.split_once('\t').unwrap_or((&line, ""));
        let phonemes: Vec<&str> = phonemes.split_whitespace().collect();
        let c = c2k.infer(word);
        let p = p2k.infer(&phonemes);
        println!(
            "{word}\t{}\t{:.6}\t{}\t{c}\t{}\t{p}\t{}",
            ngram.is_valid(word),
            ngram.score(word),
            ngram.as_is(word),
            ap.predict(&c),
            ap.predict(&p),
        );
    }
    Ok(())
}
