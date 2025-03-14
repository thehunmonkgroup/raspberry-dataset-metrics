# Llama-3.2-3B
lm-eval --model hf \
  --model_args pretrained=unsloth/Llama-3.2-3B-Instruct,dtype=float16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-llama-3.2-3b \
  --log_samples

lm-eval --model hf \
  --model_args pretrained=unsloth/Llama-3.2-3B-Instruct,peft=/home/hunmonk/git/raspberry/outputs/llama-3.2-3b/checkpoint-177,dtype=float16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-fine-tune-llama-3.2-3b \
  --log_samples

# Llama-3.1-8B
lm-eval --model hf \
  --model_args pretrained=unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit,dtype=float16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-llama-3.1-8b \
  --log_samples

lm-eval --model hf \
  --model_args pretrained=unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit,peft=/home/hunmonk/git/raspberry/outputs/llama-3.1-8b/checkpoint-177,dtype=float16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-fine-tune-llama-3.1-8b \
  --log_samples

# Mistral 7B
lm-eval --model hf \
  --model_args pretrained=unsloth/mistral-7b-instruct-v0.3-bnb-4bit,dtype=float16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-mistral-7b \
  --log_samples

lm-eval --model hf \
  --model_args pretrained=unsloth/mistral-7b-instruct-v0.3-bnb-4bit,peft=/home/hunmonk/git/raspberry/outputs/mistral-7b/checkpoint-177,dtype=float16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size 8 \
  --output_path ./results-fine-tune-mistral-7b \
  --log_samples
