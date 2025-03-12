#!/usr/bin/env python3
"""
Universal training script for fine-tuning language models using Unsloth.
Loads configuration from YAML files and abstracts model-specific logic.
"""

import argparse
import logging
import yaml
import pprint
from pathlib import Path
import sys
from typing import Any

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import (
    get_chat_template,
    standardize_sharegpt,
    # train_on_responses_only,
)
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
# from transformers import TrainingArguments, DataCollatorForSeq2Seq

from raspberry_dataset_metrics import constants
from raspberry_dataset_metrics import util
from .logger import Logger


class Trainer:
    """Main class for fine-tuning language models using configuration from YAML files."""

    def __init__(
        self, config_path: Path, dataset_file: Path, debug: bool = False
    ) -> None:
        """Initialize trainer with configuration from YAML.

        :param config_path: Path to the YAML configuration file
        :type config_path: Path
        :param dataset_file: Path to the dataset file
        :type dataset_file: Path
        """
        self.config_path: Path = config_path
        self.dataset_file: Path = dataset_file
        self.config: dict[str, Any] = self._load_config(self.config_path)
        self.log: logging.Logger = Logger(self.__class__.__name__, debug=debug)
        self.log.info(
            f"Initializing trainer with configuration: {self.config_path}, dataset: {self.dataset_file}"
        )
        self.log.debug(f"Configuration: {self.config}")

    def _load_config(self, config_path: Path) -> dict[str, Any]:
        """Load and merge configuration from YAML with defaults.

        :param config_path: Path to the YAML configuration file
        :type config_path: Path
        :return: Merged configuration dictionary
        :rtype: Dict[str, Any]
        :raises FileNotFoundError: If the config file doesn't exist
        :raises yaml.YAMLError: If the YAML file is invalid
        """
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            merged_config = {
                "max_seq_length": constants.MAX_SEQ_LENGTH,
                "dtype": constants.DTYPE,
                "load_in_4bit": constants.LOAD_IN_4BIT,
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
                "system_message": constants.SYSTEM_MESSAGE,
                "format_with_eos_token": constants.FORMAT_WITH_EOS_TOKEN,
            }
            merged_config.update(config)
            if type(merged_config["learning_rate"]) is str:
                merged_config["learning_rate"] = float(merged_config["learning_rate"])
            return merged_config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

    def _get_model_family_settings(self) -> dict[str, Any]:
        """Get model-specific settings based on the model family.

        :return: Dictionary of model-specific settings
        :rtype: Dict[str, Any]
        :raises ValueError: If model family is not supported
        """
        model_family = self.config.get("model_family")
        if not model_family or model_family not in constants.MODEL_FAMILIES:
            raise ValueError(
                "Unsupported or unspecified model family. Please specify 'model_family' in config."
            )
        return constants.MODEL_FAMILIES[model_family]

    def _transform_dataset(self, dataset: Any) -> Any:
        """Transform dataset into conversation format.

        :param dataset: Input dataset
        :type dataset: Any
        :return: Transformed dataset
        :rtype: Any
        """
        system_message = self.config.get("system_message", False)

        def transform_format(
            example: dict[str, str],
        ) -> dict[str, list[dict[str, str]]]:
            elements = []
            if system_message:
                elements.append({"from": "system", "value": system_message})
            elements.append({"from": "human", "value": example["user"]})
            elements.append({"from": "gpt", "value": example["assistant"]})
            return {"conversations": elements}

        dataset = load_dataset(
            "json", data_files=str(self.dataset_file), trust_remote_code=False
        )
        dataset = dataset["train"]
        transformed_dataset = dataset.map(transform_format)
        transformed_dataset = standardize_sharegpt(transformed_dataset)
        return transformed_dataset

    def _load_model_and_tokenizer(self) -> tuple[Any, Any]:
        """Load model and tokenizer with appropriate configuration.

        :return: Tuple of (model, tokenizer)
        :rtype: tuple
        """
        self.log.info(f"Loading model: {self.config['model_name']}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config["model_name"],
            max_seq_length=self.config["max_seq_length"],
            dtype=self.config["dtype"],
            load_in_4bit=self.config["load_in_4bit"],
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config["lora_rank"],
            target_modules=self.config["target_modules"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=self.config["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",  # pyright: ignore[reportArgumentType]
            random_state=self.config["random_seed"],
            use_rslora=self.config["use_rslora"],
            loftq_config=None,
        )
        model_settings = self._get_model_family_settings()
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=model_settings["chat_template"],
        )
        return model, tokenizer

    def train(self) -> dict[str, Any]:
        """Train the model using the loaded configuration.

        :return: Training statistics
        :rtype: Dict[str, Any]
        """
        self.log.info("Starting training process")
        model, tokenizer = self._load_model_and_tokenizer()
        dataset = self._transform_dataset(None)

        def formatting_prompts_func(examples: dict[str, Any]) -> dict[str, Any]:
            convos = examples["conversations"]
            texts = []
            for convo in convos:
                text = tokenizer.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=False
                )
                if self.config["format_with_eos_token"]:
                    text += tokenizer.eos_token
                texts.append(text)
            return {"text": texts}

        formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
        train_test_split = formatted_dataset.train_test_split(
            test_size=self.config["test_size"],
            seed=self.config["random_seed"],
        )
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
        self.log.info(
            f"Dataset prepared. Training samples: {len(train_dataset)}, Evaluation samples: {len(eval_dataset)}"
        )
        output_dir = self.config.get(
            "output_dir", f"outputs/{util.get_config_base_name(self.config_path)}"
        )
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        training_args = TrainingArguments(
            per_device_train_batch_size=self.config["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            warmup_steps=self.config["warmup_steps"],
            num_train_epochs=self.config["num_train_epochs"],
            learning_rate=self.config["learning_rate"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=self.config["logging_steps"],
            optim="adamw_8bit",
            weight_decay=self.config["weight_decay"],
            lr_scheduler_type=self.config["scheduler_type"],
            seed=self.config["random_seed"],
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,  # pyright: ignore[reportCallIssue]
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",  # pyright: ignore[reportCallIssue]
            max_seq_length=self.config[  # pyright: ignore[reportCallIssue]
                "max_seq_length"
            ],
            # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            dataset_num_proc=self.config[  # pyright: ignore[reportCallIssue]
                "dataset_num_proc"
            ],
            packing=False,  # pyright: ignore[reportCallIssue]
            args=training_args,
        )
        # model_settings = self._get_model_family_settings()
        # trainer = train_on_responses_only(
        #     trainer,
        #     instruction_part=model_settings["instruction_part"],
        #     response_part=model_settings["response_part"],
        # )
        self.log.info("Starting training")
        training_stats = trainer.train()
        self.log.info("Training completed")
        self.log.debug(f"Training stats: {training_stats}")
        return training_stats


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    :return: Parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Train language models with configurations from YAML files"
    )
    parser.add_argument("config_file", help="Path to the YAML config file")
    parser.add_argument("dataset_file", help="Path to the dataset file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    """Main entry point for the training script.

    :return: Exit code
    :rtype: int
    """
    try:
        args = parse_args()
        config_path = Path(args.config_file)
        dataset_file = Path(args.dataset_file)
        trainer = Trainer(config_path, dataset_file, args.debug)
        stats = trainer.train()
        print("\nTraining completed. Stats:")
        pprint.pprint(stats)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
