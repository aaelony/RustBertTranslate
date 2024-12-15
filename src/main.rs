//! A test of a CLI tool to translate an input sentence.
//!  cargo run -- --input-text "Hello my friends!"
//!  cargo run -- --input-text "This is fun!" --source-lang English --target-lang French
//!  cargo run -- --input-text "This is fun!" --source-lang English --target-lang Spanish

use clap::Parser;
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::translation::{Language, TranslationModelBuilder};
use tch::Device;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Opt {
    #[clap(short, long)]
    /// Input text
    input_text: Option<String>,

    #[clap(long)]
    /// Source language
    source_lang: Option<String>,

    #[clap(long)]
    /// Target language
    target_lang: Option<String>,
}

fn get_language(lang: &str) -> Language {
    match lang {
        "French" => Language::French,
        "English" => Language::English,
        "Romanian" => Language::Romanian,
        "Italian" => Language::Italian,
        "Spanish" => Language::Spanish,
        "Portuguese" => Language::Portuguese,
        _ => Language::French, // Default to French if language not recognized
    }
}

fn main() {
    let arg = Opt::parse();
    let source_lang = match arg.source_lang {
        Some(source_lang) => source_lang,
        None => "English".to_string(),
    };

    let target_lang = match arg.target_lang {
        Some(target_lang) => target_lang,
        None => "French".to_string(),
    };

    let source_language = get_language(source_lang.as_str());
    let target_language = get_language(target_lang.as_str());

    let input_text = match arg.input_text {
        Some(input_text) => input_text,
        None => "No string was provided".to_string(),
    };

    let model = TranslationModelBuilder::new()
        .with_device(Device::cuda_if_available())
        .with_model_type(ModelType::Marian)
        .with_source_languages(&[Language::English]) // Use enum directly
        .with_target_languages(&[Language::French]) // Use enum directly
        .create_model()
        .map_err(|e| format!("Failed to load translation model: {}", e));

    // let input_text = "Hello world";

    //let output = model.translate(&[input_text], None, Language::French);
    let output = model
        .expect("REASON")
        .translate(&[&input_text], source_language, target_language);
    //.translate(&[&input_text], None, Language::French);

    println!("Translation: {:?}", output);
}
