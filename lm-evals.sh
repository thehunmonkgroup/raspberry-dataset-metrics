# Llama-3.2-3B
lm-eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct,dtype=bfloat16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-llama-3.2-3b \
  --log_samples

lm-eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct,peft=/home/hunmonk/git/raspberry/outputs/llama-3.2-3b/checkpoint-225,dtype=bfloat16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-fine-tune-llama-3.2-3b \
  --log_samples

# Llama-3.1-8B
lm-eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-llama-3.1-8b \
  --log_samples

lm-eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,peft=/home/hunmonk/git/raspberry/outputs/llama-3.1-8b/checkpoint-225,dtype=bfloat16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-fine-tune-llama-3.1-8b \
  --log_samples

# Mistral 7B
lm-eval --model hf \
  --model_args pretrained=mistralai/Mistral-7B-Instruct-v0.3,dtype=bfloat16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-mistral-7b \
  --log_samples

lm-eval --model hf \
  --model_args pretrained=unsloth/Mistral-7B-Instruct-v0.3,peft=/home/hunmonk/git/raspberry/outputs/mistral-7b/checkpoint-246,dtype=bfloat16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-fine-tune-mistral-7b \
  --log_samples

# Phi-4
lm-eval --model hf \
  --model_args pretrained=microsoft/phi-4,dtype=bfloat16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-phi-4 \
  --log_samples

lm-eval --model hf \
  --model_args pretrained=microsoft/phi-4,peft=/home/hunmonk/git/raspberry/outputs/phi-4/checkpoint-213,dtype=bfloat16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-fine-tune-phi-4 \
  --log_samples

# Qwen-2.5 7B
lm-eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,dtype=bfloat16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-qwen-2.5-7b \
  --log_samples

lm-eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,peft=/home/hunmonk/git/raspberry/outputs/qwen-2.5-7b/checkpoint-225,dtype=bfloat16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-fine-tune-qwen-2.5-7b \
  --log_samples

# Gemma 3
lm-eval --model hf \
  --model_args pretrained=google/gemma-3-4b-it,dtype=bfloat16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-gemma-3-4b-it \
  --log_samples

lm-eval --model hf \
  --model_args pretrained=google/gemma-3-4b-it,peft=/home/hunmonk/git/raspberry/outputs/gemma-3-4b-it/checkpoint-219,dtype=bfloat16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-fine-tune-gemma-3-4b-it \
  --log_samples
