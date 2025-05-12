# Llama-3.2-3B
lm-eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct,dtype=bfloat16 \
  --tasks mmlu,gsm8k \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-llama-3.2-3b \
  --log_samples

lm-eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct,peft=thehunmonkgroup/llama-3.2-3b-2025_05_11_17_34,tokenizer=thehunmonkgroup/llama-3.2-3b-2025_05_11_17_34,dtype=bfloat16 \
  --tasks mmlu,gsm8k \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-fine-tune-llama-3.2-3b \
  --log_samples

# Llama-3.1-8B
lm-eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16 \
  --tasks mmlu,gsm8k \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-llama-3.1-8b \
  --log_samples

lm-eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,peft=thehunmonkgroup/llama-3.1-8b-2025_05_11_18_16,tokenizer=thehunmonkgroup/llama-3.1-8b-2025_05_11_18_16,dtype=bfloat16 \
  --tasks mmlu,gsm8k \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-fine-tune-llama-3.1-8b \
  --log_samples

# Mistral 7B
lm-eval --model hf \
  --model_args pretrained=mistralai/Mistral-7B-Instruct-v0.3,dtype=bfloat16 \
  --tasks mmlu,gsm8k \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-mistral-7b \
  --log_samples

lm-eval --model hf \
  --model_args pretrained=mistralai/Mistral-7B-Instruct-v0.3,peft=thehunmonkgroup/mistral-7b-2025_05_11_18_42,tokenizer=thehunmonkgroup/mistral-7b-2025_05_11_18_42,dtype=bfloat16 \
  --tasks mmlu,gsm8k \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-fine-tune-mistral-7b \
  --log_samples

# Phi-4
lm-eval --model hf \
  --model_args pretrained=microsoft/phi-4,dtype=bfloat16 \
  --tasks mmlu,gsm8k \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-phi-4 \
  --log_samples

lm-eval --model hf \
  --model_args pretrained=microsoft/phi-4,peft=thehunmonkgroup/phi-4-2025_05_11_21_01,tokenizer=thehunmonkgroup/phi-4-2025_05_11_21_01,dtype=bfloat16 \
  --tasks mmlu,gsm8k \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-fine-tune-phi-4 \
  --log_samples

# Qwen-2.5 7B
lm-eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,dtype=bfloat16 \
  --tasks mmlu,gsm8k \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-qwen-2.5-7b \
  --log_samples

lm-eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,peft=thehunmonkgroup/qwen-2.5-7b-2025_05_11_20_12,tokenizer=thehunmonkgroup/qwen-2.5-7b-2025_05_11_20_12,dtype=bfloat16 \
  --tasks mmlu,gsm8k \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-fine-tune-qwen-2.5-7b \
  --log_samples

# Gemma 3
lm-eval --model hf \
  --model_args pretrained=google/gemma-3-4b-it,dtype=bfloat16 \
  --tasks mmlu,gsm8k \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-gemma-3-4b-it \
  --log_samples

# For now, requires using locally cached files because Gemma 3 model config needs to be hacked to include proper vocab_size.
lm-eval --model hf \
  --model_args pretrained=google/gemma-3-4b-it,peft=thehunmonkgroup/gemma-4b-it-2025_05_11_22_00,tokenizer=thehunmonkgroup/gemma-4b-it-2025_05_11_22_00,dtype=bfloat16,local_files_only=True \
  --tasks mmlu,gsm8k \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-fine-tune-gemma-3-4b-it \
  --log_samples
