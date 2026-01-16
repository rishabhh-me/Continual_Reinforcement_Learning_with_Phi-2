import torch
import argparse
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel

def train_rlhf(args):
    # Config
    config = PPOConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_ppo_epochs=4,
        seed=42,
        use_cpu=not torch.cuda.is_available(),
        output_dir=args.output_dir
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    # Dataset
    print(f"Loading data from {args.train_file}")
    def build_dataset(data_path):
        ds = load_dataset("json", data_files=data_path, split="train")
        def tokenize(sample):
            sample["input_ids"] = tokenizer.encode(sample["prompt"])
            return sample
        ds = ds.map(tokenize, batched=False)
        return ds

    dataset = build_dataset(args.train_file)
    
    # Model (Policy)
    print("Loading Policy Model...")
    bnb_config = None
    if args.load_in_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )
    model.config.use_cache = False 
    
    if args.dpo_model_dir:
        print(f"Loading DPO adapter from {args.dpo_model_dir}")
        try:
            model = PeftModel.from_pretrained(model, args.dpo_model_dir)
            print("Adapter loaded.")
        except Exception as e:
            print(f"Failed to load adapter: {e}")

    # Reward Model
    print("Loading Reward Model...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path, 
        num_labels=1,
        device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    if reward_model.config.pad_token_id is None:
        reward_model.config.pad_token_id = tokenizer.pad_token_id

    # Value Model
    print("Loading Value Model...")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path, # Init from RM weights as a good starting point
        num_labels=1,
        device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    if value_model.config.pad_token_id is None:
        value_model.config.pad_token_id = tokenizer.pad_token_id

    print("Initializing PPOTrainer...")
    trainer = PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=model,
        ref_model=None, 
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
    )

    print("Starting PPO Training...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--dpo_model_dir", type=str, default="results_dpo")
    parser.add_argument("--reward_model_path", type=str, default="results_rm")
    parser.add_argument("--train_file", type=str, default="data/dpo_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="results_rlhf")
    parser.add_argument("--lr", type=float, default=1.41e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--save_steps", type=int, default=10)
    
    args = parser.parse_args()
    train_rlhf(args)
