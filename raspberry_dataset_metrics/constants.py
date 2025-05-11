#!/usr/bin/env python3
"""
Default configuration values for model training.
These can be overridden by values in config files.
"""

# Model parameters
MAX_SEQ_LENGTH = 1024  # Context window size
LOAD_IN_4BIT = True  # Whether to use 4-bit quantization
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True
DEVICE_MAP = "auto"
#
# Training hyperparameters
LORA_RANK = 256
LORA_ALPHA = 256
LORA_DROPOUT = 0.1
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
# RANDOM_SEED = 3407
# USE_RSLORA = False
#
# Training parameters
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_RATIO = 0.03
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.001
LOGGING_STEPS = 1
# DATASET_NUM_PROC = 2
OPTIMIZER_TYPE = "paged_adamw_32bit"
SCHEDULER_TYPE = "cosine"
TRAIN_TEST_SPLIT_SIZE = 0.1
FORMAT_WITH_EOS_TOKEN = False
ENABLE_PACKING = True
MAX_GRAD_NORM = 0.3
GROUP_BY_LENGTH = False
GRADIENT_CHECKPOINTING = True
SAVE_STEPS = 0

# System message
SYSTEM_MESSAGE = "You are a helpful assistant."
# SYSTEM_MESSAGE = """
# You are a reasoning agent that uses chain-of-thought reasoning to solve problems and answer queries. Always structure your response in two parts: your step-by-step reasoning wrapped in <reasoning></reasoning> tags, followed by your final answer wrapped in <output></output> tags.
#
# For example:
#
# User: Why might increasing atmospheric CO2 lead to ocean acidification?
#
# A:
#
# <reasoning>
# 1. CO2 from the atmosphere dissolves in seawater
# 2. When dissolved, CO2 reacts with H2O to form carbonic acid (H2CO3)
# 3. H2CO3 dissociates into H+ and HCO3- ions
# 4. The increase in H+ ions directly decreases ocean pH
# 5. This process forms a feedback loop: more atmospheric CO2 leads to more dissolved CO2, producing more H+ ions
# </reasoning>
#
# <output>
# Ocean acidification occurs because atmospheric CO2 dissolves in seawater and undergoes chemical reactions that increase the concentration of hydrogen ions, directly lowering the ocean's pH.
# </output>
# """

# Generation parameters
TEMPERATURE = 0.8
MIN_P = None
MAX_NEW_TOKENS = 1024

# Model-specific settings
MODEL_FAMILIES = {
    "phi-4": {
        "attn_implementation": "flash_attention_2",
        "instruction_part": "<|im_start|>user<|im_sep|>",
        "response_part": "<|im_start|>assistant<|im_sep|>",
        "response_extraction_pattern": r".*<\|im_start\|>assistant<\|im_sep\|>([\s\S]*?)<\|im_end\|>$",
    },
    "llama-3.1": {
        "attn_implementation": "sdpa",
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "response_extraction_pattern": r".*<\|start_header_id\|>assistant<\|end_header_id\|>\n\n([\s\S]*?)<\|eot_id\|>$",
    },
    "mistral": {
        "attn_implementation": "eager",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "response_extraction_pattern": r"\[/INST\]\s*([\s\S]*?)</s>$",
    },
    "qwen-2.5": {
        "attn_implementation": "flash_attention_2",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "response_extraction_pattern": r".*<\|im_start\|>assistant\n([\s\S]*?)<\|im_end\|>$",
    },
    "qwen-3": {
        "attn_implementation": "flash_attention_2",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "response_extraction_pattern": r".*<\|im_start\|>assistant\n([\s\S]*?)<\|im_end\|>$",
    },
    "gemma-3": {
        "attn_implementation": "eager",
        "instruction_part": "<start_of_turn>user\n",
        "response_part": "<start_of_turn>model\n",
        "response_extraction_pattern": r".*<start_of_turn>model\n([\s\S]*?)$",
    },
}
