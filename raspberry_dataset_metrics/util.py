from pathlib import Path
import yaml
from typing import Any


def get_config_base_name(config_path: Path | str) -> str:
    """Get the base name of a config file without extension.

    :param config_path: Path to the config file
    :type config_path: Path | str
    :return: Base name without extension
    :rtype: str
    """
    return Path(config_path).stem


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config and merge with default values.

    :param config_path: Path to the YAML config file
    :type config_path: Path
    :return: Config dictionary with defaults applied
    :rtype: dict[str, Any]
    :raises FileNotFoundError: If config file doesn't exist
    :raises yaml.YAMLError: If YAML is invalid
    """
    from raspberry_dataset_metrics import constants

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Define default values for all possible configuration parameters
    default_config = {
        # Common parameters
        "max_seq_length": constants.MAX_SEQ_LENGTH,
        "dtype": constants.DTYPE,
        "load_in_4bit": constants.LOAD_IN_4BIT,
        "system_message": constants.SYSTEM_MESSAGE,

        # Training parameters
        "lora_rank": constants.LORA_RANK,
        "lora_alpha": constants.LORA_ALPHA,
        "lora_dropout": constants.LORA_DROPOUT,
        "target_modules": constants.TARGET_MODULES,
        "random_seed": constants.RANDOM_SEED,
        "use_rslora": constants.USE_RSLORA,
        "per_device_train_batch_size": constants.PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": constants.GRADIENT_ACCUMULATION_STEPS,
        "warmup_steps": constants.WARMUP_STEPS,
        "num_train_epochs": constants.NUM_TRAIN_EPOCHS,
        "learning_rate": constants.LEARNING_RATE,
        "weight_decay": constants.WEIGHT_DECAY,
        "logging_steps": constants.LOGGING_STEPS,
        "dataset_num_proc": constants.DATASET_NUM_PROC,
        "scheduler_type": constants.SCHEDULER_TYPE,
        "test_size": constants.TRAIN_TEST_SPLIT_SIZE,
        "format_with_eos_token": constants.FORMAT_WITH_EOS_TOKEN,

        # Chat parameters
        "temperature": constants.TEMPERATURE,
        "min_p": constants.MIN_P,
        "max_new_tokens": constants.MAX_NEW_TOKENS,
    }

    # Merge defaults with provided config
    merged_config = default_config.copy()
    merged_config.update(config)

    # Handle learning_rate as string
    if type(merged_config["learning_rate"]) is str:
        merged_config["learning_rate"] = float(merged_config["learning_rate"])

    return merged_config
