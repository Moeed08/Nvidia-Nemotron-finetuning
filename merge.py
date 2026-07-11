from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "nemotron-finetuned",
    max_seq_length = 1024,
    load_in_4bit = True,      # <-- change this back to True
    trust_remote_code = True,
    attn_implementation = "eager",
)

model.save_pretrained_merged("nemotron-merged", tokenizer, save_method = "merged_16bit")