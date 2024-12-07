#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
import sys
import pprint
import torch
from transformers import TextStreamer

base_model_name = "unsloth/Llama-3.2-3B-Instruct"
max_seq_length = 32768  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.


ASSISTANT_RESPONSE_EXTRACTION_PATTERN = r".*<\|start_header_id\|>assistant<\|end_header_id\|>\n\n([\s\S]*?)<\|eot_id\|>$"


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


class CaptureTextStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=False):
        super().__init__(tokenizer, skip_prompt)
        self.captured_text = []

    def put(self, value):
        if torch.is_tensor(value):
            value = value.cpu()
            text = self.tokenizer.decode(value[0] if value.dim() > 1 else value)
            display_text = text.replace("<|eot_id|>", "")  # Remove tag for display only
            if display_text.strip():  # Only display if there's content
                super().put(value)  # Pass the original tensor to super().put()
            if text.strip():  # Always append original text (with tag) to captured_text
                self.captured_text.append(text)
        else:
            super().put(value)


def chat(checkpoint_path):
    from unsloth import FastLanguageModel
    from peft import PeftModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    print(f"Loading adapter weights from {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)

    # Convert to inference mode.
    model = FastLanguageModel.for_inference(model)

    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("\nChat interface started (Press Ctrl+C to exit)")

    # Initialize conversation history
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    while True:
        try:
            user_input = input("\nYou: ")
            print("\nAssistant:\n")
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,  # Must add for generation
                return_tensors="pt",
                return_dict=True,
            ).to("cuda")
            text_streamer = CaptureTextStreamer(tokenizer, skip_prompt=True)
            eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            _ = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                streamer=text_streamer,
                max_new_tokens=1024,
                eos_token_id=eos_token_id,
                use_cache=True,
                temperature=0.1,
                min_p=0.1,
            )
            response = "".join(text_streamer.captured_text)
            # print("############")
            # print(response)
            # print("############")

            # Create prompt with full conversation history
            match = re.search(ASSISTANT_RESPONSE_EXTRACTION_PATTERN, response, re.DOTALL)
            if match:
                assistant_response = match.group(1).strip()
                # print("############")
                # print(assistant_response)
                # print("############")
                messages.append({"role": "assistant", "content": assistant_response})
            else:
                print("WARNING: Could not extract assistant response from:", response)
            # pprint.pprint(messages)
        except KeyboardInterrupt:
            print('\nExiting chat interface')
            sys.exit(0)

if __name__ == "__main__":
    checkpoint_path = parse_args()
    chat(checkpoint_path)
