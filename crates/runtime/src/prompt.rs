use crate::{DlirError, PromptTemplate, Result};

pub fn render_prompt(template: PromptTemplate, prompt: &str) -> Result<String> {
    if prompt.trim().is_empty() {
        return Err(DlirError::EmptyPrompt);
    }
    Ok(template.source().replace("{prompt}", prompt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SupportedModelId, artifacts::ArtifactRepository};
    use tokenizers::Tokenizer;

    #[test]
    fn renders_smol_template_exactly() {
        assert_eq!(
            render_prompt(PromptTemplate::SmolChatMl, "Hello").unwrap(),
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn renders_tinyllama_template_exactly() {
        assert_eq!(
            render_prompt(PromptTemplate::TinyLlamaChat, "Hello").unwrap(),
            "<|user|>\nHello</s>\n<|assistant|>\n"
        );
    }

    fn pinned_token_ids(model: SupportedModelId) -> Vec<u32> {
        let spec = model.spec();
        let repository = ArtifactRepository::new(spec).unwrap();
        let metadata = repository.download_metadata().unwrap();
        let tokenizer = Tokenizer::from_file(metadata.tokenizer).unwrap();
        let rendered = render_prompt(spec.prompt_template, "Hello").unwrap();
        tokenizer
            .encode(rendered, false)
            .unwrap()
            .get_ids()
            .to_vec()
    }

    #[test]
    #[ignore = "downloads the pinned SmolLM2 tokenizer metadata"]
    fn smollm2_chat_template_token_ids_are_golden() {
        assert_eq!(
            pinned_token_ids(SupportedModelId::SmolLm2_135MInstruct),
            vec![
                1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018,
                411, 407, 19712, 8182, 2, 198, 1, 4093, 198, 19556, 2, 198, 1, 520, 9531, 198,
            ]
        );
    }

    #[test]
    #[ignore = "downloads the pinned TinyLlama tokenizer metadata"]
    fn tinyllama_chat_template_token_ids_are_golden_without_added_bos() {
        let ids = pinned_token_ids(SupportedModelId::TinyLlama1_1BChat);
        assert_eq!(
            ids,
            vec![
                529, 29989, 1792, 29989, 29958, 13, 10994, 2, 29871, 13, 29966, 29989, 465, 22137,
                29989, 29958, 13,
            ]
        );
        assert_ne!(ids.first(), Some(&1), "BOS must not be duplicated");
    }
}
