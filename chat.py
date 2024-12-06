#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chat interface for Unsloth-trained language models. Loads a trained model from "
                    "a checkpoint directory and starts an interactive chat session."
    )
    parser.add_argument(
        "checkpoint_dir",
        help="Name of the checkpoint directory within 'outputs' (e.g., 'checkpoint-60')"
    )
    args = parser.parse_args()

    # Construct and validate path
    checkpoint_path = Path("outputs") / args.checkpoint_dir
    if not checkpoint_path.exists():
        parser.error(f"Checkpoint directory not found: {checkpoint_path}")
    if not (checkpoint_path / "adapter_model.safetensors").exists():
        parser.error(f"No adapter model found in {checkpoint_path}")

    return checkpoint_path

def chat(checkpoint_path):
    import torch
    from unsloth import FastLanguageModel
    from peft import PeftModel
    # Load the base model and tokenizer
    base_model_name = "unsloth/llama-3-8b-bnb-4bit"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"Loading adapter weights from {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)

    # Convert to inference mode.
    model = FastLanguageModel.for_inference(model)

    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("\nChat interface started (Press Ctrl+C to exit)")

    while True:
        try:
            user_input = input("\nYou: ")
            prompt = f"Human: {user_input}\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt")
            # Move input tensors to same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Assistant:")[-1].strip()
            print("\nAssistant:", response)

        except KeyboardInterrupt:
            print('\nExiting chat interface...')
            sys.exit(0)

if __name__ == "__main__":
    checkpoint_path = parse_args()
    chat(checkpoint_path)
