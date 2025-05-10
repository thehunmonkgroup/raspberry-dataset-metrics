#!/usr/bin/env python3
"""
Base class for model handling, providing shared functionality for training and chat interfaces.
"""

import logging
import yaml
from pathlib import Path
from typing import Any
import torch

from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)

from raspberry_dataset_metrics import constants
from raspberry_dataset_metrics.logger import Logger
from raspberry_dataset_metrics import util


class BaseModelHandler:
    """Base class with shared functionality for training and chat interfaces."""

    def __init__(
        self, config_path: Path, debug: bool = False
    ) -> None:
        """Initialize with configuration from YAML.

        :param config_path: Path to the YAML configuration file
        :type config_path: Path
        :param debug: Enable debug logging
        :type debug: bool
        """
        self.config_path: Path = config_path
        self.config: dict[str, Any] = self._load_config(self.config_path)
        self.log: logging.Logger = Logger(self.__class__.__name__, debug=debug)
        self.log.info(f"Initializing with configuration: {self.config_path}")
        self.log.debug(f"Configuration: {self.config}")
        self.model_settings: dict[str, Any] = self._get_model_family_settings()
        self.log.debug(f"Loaded model faily settings: {self.model_settings}")
        self._login_to_huggingface_hub()

    def _load_config(self, config_path: Path) -> dict[str, Any]:
        """Load and merge configuration from YAML with defaults.

        :param config_path: Path to the YAML configuration file
        :type config_path: Path
        :return: Merged configuration dictionary
        :rtype: dict[str, Any]
        :raises FileNotFoundError: If the config file doesn't exist
        :raises yaml.YAMLError: If the YAML file is invalid
        """
        try:
            return util.load_yaml_config(config_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

    def _get_model_family_settings(self) -> dict[str, Any]:
        """Get model-specific settings based on the model family.

        :return: Dictionary of model-specific settings
        :rtype: dict[str, Any]
        :raises ValueError: If model family is not supported
        """
        model_family = self.config.get("model_family")
        if not model_family or model_family not in constants.MODEL_FAMILIES:
            raise ValueError(
                "Unsupported or unspecified model family. Please specify 'model_family' in config."
            )
        return constants.MODEL_FAMILIES[model_family].copy()

    def _login_to_huggingface_hub(self):
        self.log.debug("Logging in to Hugging Face Hub")
        util.login_to_huggingface_hub()

    def has_existing_peft_config(self, model: Any) -> bool:
        has_peft_config = hasattr(model, "peft_config")
        if has_peft_config:
            self.log.warning(f"Model already has default PEFT configuration: {model.peft_config}")
        else:
            self.log.debug("Model has no existing PEFT configuration.")
        return has_peft_config

    def is_model_loaded_in_4bit(self, model: Any) -> bool:
        if model.is_loaded_in_4bit:
            self.log.info("Model is loaded in 4-bit mode.")
        else:
            self.log.info("Model is not loaded in 4-bit mode.")
        return model.is_loaded_in_4bit

    def load_model_and_tokenizer(self, pad_token: str = "<pad>") -> tuple[Any, Any]:
        """Load model and tokenizer with appropriate configuration.

        :return: Tuple of (model, tokenizer)
        :rtype: tuple[Any, Any]
        """
        self.log.info(f"Loading model: {self.config['model_name']}")
        bnb_4bit_compute_dtype = getattr(torch, self.config["bnb_4bit_compute_dtype"])
        bnb_setup = BitsAndBytesConfig(load_in_4bit = self.config['load_in_4bit'],
                                    bnb_4bit_quant_type = self.config['bnb_4bit_quant_type'],
                                    bnb_4bit_use_double_quant = self.config['bnb_4bit_use_double_quant'],
                                    bnb_4bit_compute_dtype = bnb_4bit_compute_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            quantization_config = bnb_setup,
            device_map = self.config["device_map"],
        )
        self.has_existing_peft_config(model)
        self.is_model_loaded_in_4bit(model)
        self.log.info(f"Loading tokenizer: {self.config['model_name']}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"],
            use_fast=True,
            trust_remote_code = True,
        )
        if tokenizer.pad_token is None:
            self.log.info(f"Adding pad token: {pad_token}")
            tokenizer.add_special_tokens({"pad_token": pad_token})
            pad_id = tokenizer.pad_token_id
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = pad_id
        return model, tokenizer

    def generate_prompt(self, tokenizer: Any, chat: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,  # <|start_header_id|>assistant … tag
        )
        return prompt
