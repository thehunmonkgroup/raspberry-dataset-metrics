#!/usr/bin/env python3

import pprint
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

max_seq_length = 32768  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

SYSTEM_MESSAGE = """
You are a reasoning agent that uses chain-of-thought reasoning to solve problems and answer queries. Always structure your response in two parts: your step-by-step reasoning wrapped in <reasoning></reasoning> tags, followed by your final answer wrapped in <output></output> tags.

For example:

User: Why might increasing atmospheric CO2 lead to ocean acidification?

Assistant:

<reasoning>
1. CO2 from the atmosphere dissolves in seawater
2. When dissolved, CO2 reacts with H2O to form carbonic acid (H2CO3)
3. H2CO3 dissociates into H+ and HCO3- ions
4. The increase in H+ ions directly decreases ocean pH
5. This process forms a feedback loop: more atmospheric CO2 leads to more dissolved CO2, producing more H+ ions
</reasoning>

<output>
Ocean acidification occurs because atmospheric CO2 dissolves in seawater and undergoes chemical reactions that increase the concentration of hydrogen ions, directly lowering the ocean's pH.
</output>
"""


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {
        "text": texts,
    }


model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name="unsloth/Llama-3.2-3B-Instruct",  # More models at https://huggingface.co/unsloth
    # model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",  # More models at https://huggingface.co/unsloth
    model_name="unsloth/phi-4-unsloth-bnb-4bit",  # More models at https://huggingface.co/unsloth
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but=0 is optimized
    bias="none",  # Supports any, but="none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="phi-4",
)

dataset = load_dataset("json", data_files="training-data-consolidated-final.jsonl")
def transform_format(example):
    return {
        'conversations': [
            {'from': 'system', 'value': SYSTEM_MESSAGE},
            {'from': 'human', 'value': example['user']},
            {'from': 'gpt', 'value': example['assistant']}
        ]
    }
dataset = dataset.map(transform_format)
dataset = standardize_sharegpt(dataset["train"])
dataset = dataset.map(formatting_prompts_func, batched=True,)
train_test_split = dataset.train_test_split(test_size=0.1, seed=3407)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']
# import pprint
# pprint.pprint(train_dataset[0])
# raise Exception("STOP")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        # max_steps=1000,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="phi_4",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user<|im_sep|>",
    response_part="<|im_start|>assistant<|im_sep|>",
)

trainer_stats = trainer.train()
pprint.pprint(trainer_stats)
