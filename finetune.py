from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/NVIDIA-Nemotron-3-Nano-4B",
    max_seq_length = 1024,
    load_in_4bit = True,
    trust_remote_code = True,
    attn_implementation = "eager"
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                       "gate_proj","up_proj","down_proj",
                       "in_proj","out_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

model.print_trainable_parameters()   # <-- add this too, quick sanity check

from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

def format_example(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
    }

dataset = dataset.map(format_example, remove_columns=dataset.column_names)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(                    # <-- THIS block is what you're replacing
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        max_seq_length = 1024,
        num_train_epochs = 5,
        fp16 = False, bf16 = False,
        output_dir = "outputs",
        padding_free = False,            # <-- new
        logging_steps = 1,               # <-- new
    ),
)
trainer.train()
model.save_pretrained("nemotron-finetuned")