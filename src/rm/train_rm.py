import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

def train_rm(args):
    print(f"Loading tokenizer {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.eos_token is None:
        # For BERT-like models
        if tokenizer.sep_token:
            tokenizer.eos_token = tokenizer.sep_token
        elif tokenizer.cls_token:
            tokenizer.eos_token = tokenizer.cls_token
        else:
             tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load raw dataset
    print(f"Loading data from {args.train_file} and {args.eval_file}")
    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_file, split="train")
    
    # Preprocessing
    def preprocess_function(examples):
        new_examples = {
            "chosen": [],
            "rejected": [],
        }
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            # Format: "Prompt\nSubgoal"
            new_examples["chosen"].append(prompt + "\n" + chosen)
            new_examples["rejected"].append(prompt + "\n" + rejected)
            
        return new_examples

    print("Preprocessing datasets...")
    # We keep chosen/rejected as text. RewardTrainer handles tokenization.
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["prompt"])
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=["prompt"])
    
    # Model
    print("Loading model")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
    
    training_args = RewardConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        report_to="tensorboard",
        use_cpu=not torch.cuda.is_available()
    )
    
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Using gpt2 because trl RewardTrainer passes use_cache=False which breaks BERT
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--train_file", type=str, default="data/dpo_train.jsonl")
    parser.add_argument("--eval_file", type=str, default="data/dpo_val.jsonl")
    parser.add_argument("--output_dir", type=str, default="results_rm")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    
    args = parser.parse_args()
    train_rm(args)
