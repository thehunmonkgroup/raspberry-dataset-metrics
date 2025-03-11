#!/usr/bin/env python3
"""
Default configuration values for model training.
These can be overridden by values in config files.
"""

# Model parameters
MAX_SEQ_LENGTH = 32768  # Context window size
DTYPE = None  # Auto-detect (Float16 for T4/V100, BFloat16 for Ampere+)
LOAD_IN_4BIT = True  # Whether to use 4-bit quantization

# Training hyperparameters
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
RANDOM_SEED = 3407
USE_RSLORA = False

# Training arguments
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 100
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
DATASET_NUM_PROC = 2
SCHEDULER_TYPE = "cosine"
TRAIN_TEST_SPLIT_SIZE = 0.1

# System message
SYSTEM_MESSAGE = """
You are a reasoning agent that uses chain-of-thought reasoning to solve problems and answer queries. Always structure your response in two parts: your step-by-step reasoning wrapped in <reasoning></reasoning> tags, followed by your final answer wrapped in <output></output> tags.

For example:

User: Why might increasing atmospheric CO2 lead to ocean acidification?

A:

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

# Generation parameters
TEMPERATURE = 0.1
MIN_P = 0.1
MAX_NEW_TOKENS = 1024

# Model-specific settings
MODEL_FAMILIES = {
    "phi-4": {
        "chat_template": "phi-4",
        "instruction_part": "<|im_start|>user<|im_sep|>",
        "response_part": "<|im_start|>assistant<|im_sep|>"
    },
    "llama-3.1": {
        "chat_template": "llama-3.1",
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n"
    },
    "qwen-2.5": {
        "chat_template": "qwen2.5",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n"
    }
}
