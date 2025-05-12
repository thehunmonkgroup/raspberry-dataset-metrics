#!/usr/bin/env python3
"""
Universal training script for fine-tuning language models using Unsloth.
Loads configuration from YAML files and abstracts model-specific logic.
"""

import os
import argparse
import pprint
from pathlib import Path
import sys
from typing import Any

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback, AutoConfig
from huggingface_hub import snapshot_download

from raspberry_dataset_metrics import util
from raspberry_dataset_metrics.base_model import BaseModelHandler


class SaveTokenizerCallback(TrainerCallback):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        # same folder Trainer just saved the model into
        ckpt_dir = os.path.join(args.output_dir, f"{state.best_model_checkpoint or f'checkpoint-{state.global_step}'}")
        self.tokenizer.save_pretrained(ckpt_dir)
        return control


class Trainer(BaseModelHandler):
    """Main class for fine-tuning language models using configuration from YAML files."""

    def __init__(
        self, config_path: Path, dataset_file: Path, debug: bool = False
    ) -> None:
        """Initialize trainer with configuration from YAML.

        :param config_path: Path to the YAML configuration file
        :type config_path: Path
        :param dataset_file: Path to the dataset file
        :type dataset_file: Path
        :param debug: Enable debug logging
        :type debug: bool
        """
        super().__init__(config_path, debug)
        self.dataset_file: Path = dataset_file
        self.log.info(f"Initializing trainer with dataset: {self.dataset_file}")
        self.model: Any = None
        self.tokenizer: Any = None
        self.peft_setup: Any = None

    def _transform_dataset(self, dataset: Any) -> Any:
        """Transform dataset into conversation format.

        :param dataset: Input dataset
        :type dataset: Any
        :return: Transformed dataset
        :rtype: Any
        """

        def transform_format(
            example: dict[str, str],
        ) -> dict[str, str]:
            chat = [
                {"role": "user",   "content": example["user"]},
                {"role": "assistant", "content": example["assistant"]}
            ]
            example["text"] = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False,
            )
            return {
                "text": example["text"],
            }

        dataset_file = str(self.dataset_file)
        self.log.debug(f"Loading dataset: {dataset_file}")
        dataset = load_dataset(
            "json",
            data_files=str(dataset_file),
            trust_remote_code=False,
            split = "train",
        )
        self.log.debug("Transforming dataset")
        training_data = dataset.map(transform_format, remove_columns=dataset.column_names)
        return training_data

    def _load_model_and_tokenizer(self) -> None:
        """Load model and tokenizer with appropriate configuration.

        :return: Tuple of (model, tokenizer)
        :rtype: tuple
        """
        self.log.debug("Loading model and tokenizer")
        model, tokenizer = self.load_model_and_tokenizer()
        peft_setup = LoraConfig(
            r=self.config["lora_rank"],
            target_modules=self.config["target_modules"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=self.config["lora_dropout"],
            bias = "none",
            task_type = "CAUSAL_LM",
            inference_mode=False,
        )
        model_family_stub = self.get_model_family_stub()
        model_adjustments = getattr(self, f"adjust_model_{model_family_stub}", None)
        if model_adjustments:
            self.log.info(f"Applying model adjustments for family {self.config['model_family']}")
            model = model_adjustments(model)
        tokenizer_adjustments = getattr(self, f"adjust_tokenizer_{model_family_stub}", None)
        if tokenizer_adjustments:
            self.log.info(f"Applying tokenizer adjustments for family {self.config['model_family']}")
            tokenizer = tokenizer_adjustments(tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.peft_setup = peft_setup

    def get_gpu_capabilities(self) -> tuple[bool, bool]:
        fp16_supported = torch.cuda.get_device_capability()[0] >= 5 and torch.cuda.is_available()
        bf16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        use_fp16 = False
        use_bf16 = False
        if bf16_supported:
            use_bf16 = True
        elif fp16_supported:
            use_fp16 = True
        self.log.debug(f"GPU capabilities: fp16={use_fp16}, bf16={use_bf16}")
        return use_fp16, use_bf16

    def train(self) -> dict[str, Any]:
        """Train the model using the loaded configuration.

        :return: Training statistics
        :rtype: dict[str, Any]
        """
        self.log.info("Starting training process")
        self._load_model_and_tokenizer()
        train_dataset = self._transform_dataset(None)
        self.log.info(
            f"Dataset prepared. Training samples: {len(train_dataset)}"
        )
        output_dir = self.config.get(
            "output_dir", f"outputs/{util.get_config_base_name(self.config_path)}"
        )
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        self.log.info(f"Results will be written to: {output_dir}")


        self.model.enable_input_require_grads() # A quirk of LoRA + gradient checkpointing
        self.model = get_peft_model(self.model, self.peft_setup)

        use_fp16, use_bf16 = self.get_gpu_capabilities()
        train_args = SFTConfig(
            output_dir = output_dir,
            num_train_epochs=self.config["num_train_epochs"],
            per_device_train_batch_size=self.config["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            learning_rate=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            optim=self.config["optimizer_type"],
            save_steps=self.config["save_steps"],
            logging_steps=self.config["logging_steps"],
            fp16 = use_fp16,
            bf16 = use_bf16,
            max_grad_norm=self.config["max_grad_norm"],
            warmup_ratio=self.config["warmup_ratio"],
            group_by_length=self.config["group_by_length"],
            lr_scheduler_type=self.config["scheduler_type"],
            gradient_checkpointing=self.config["gradient_checkpointing"],
            dataset_text_field = "text",
            max_seq_length=self.config["max_seq_length"],
            packing=self.config["enable_packing"],
            # load_best_model_at_end=True,
            # eval_strategy="no",
            # metric_for_best_model="loss",
            save_strategy="no",
        )
        trainer = SFTTrainer(
            model = self.model,
            args = train_args,
            train_dataset = train_dataset,
            processing_class=self.tokenizer,
            callbacks=[SaveTokenizerCallback(self.tokenizer)],
        )
        self.model.config.use_cache = False       # free key/value cache
        self.model.gradient_checkpointing_enable()  # discard activations
        self.log.info("Starting training")
        training_stats = trainer.train()
        trainer.save_model(output_dir)
        self.log.info("Training completed")
        model_family_stub = self.get_model_family_stub()
        post_training_adjustments = getattr(self, f"post_training_adjustment_{model_family_stub}", None)
        if post_training_adjustments:
            self.log.info(f"Applying post-training adjustments for family {self.config['model_family']}")
            post_training_adjustments()
        self.log.debug(f"Training stats: {training_stats}")
        return training_stats

    def adjust_model_llama_3_1(self, model: Any) -> Any:
        model.config.use_case = False
        model.config.pretraining_tp = 1
        return model

    def adjust_tokenizer_llama_3_1(self, tokenizer: Any) -> Any:
        tokenizer.padding_side = "left"
        return tokenizer

    def post_training_adjustment_gemma_3(self) -> None:
        vocab_size = len(self.tokenizer)
        config = AutoConfig.from_pretrained(
            self.config["model_name"],
            local_files_only=True,
        )
        config.vocab_size = vocab_size
        snapshot_dir = snapshot_download(
            repo_id=self.config["model_name"],
            local_files_only=True,
        )
        config.save_pretrained(snapshot_dir)


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
