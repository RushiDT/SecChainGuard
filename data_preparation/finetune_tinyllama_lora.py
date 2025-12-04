import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = "data/iot_blockchain_ml.jsonl"
OUTPUT_DIR = "fine_tuned_models/tinyllama-iot-sec-lora"


def load_iot_dataset():
    ds = load_dataset("json", data_files=DATASET_PATH)
    return ds["train"]


def format_example(ex):
    """
    Expected JSONL format per line:
    {
        "instruction": "...",
        "input": "...",
        "output": "..."
    }
    """
    instr = ex.get("instruction", "").strip()
    inp = ex.get("input", "").strip()
    out = ex.get("output", "").strip()

    if inp:
        prompt = f"Instruction: {instr}\nInput: {inp}\nAnswer:"
    else:
        prompt = f"Instruction: {instr}\nAnswer:"

    full_text = prompt + " " + out
    return {"text": full_text}


def main():
    print("Loading dataset...")
    raw_ds = load_iot_dataset()
    ds = raw_ds.map(format_example)

    print(f"Loading tokenizer and base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tokenizer))


    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
        )
        # For causal LM, labels are just the same as input_ids
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = ds.map(tokenize_function, remove_columns=ds.column_names)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapters to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
