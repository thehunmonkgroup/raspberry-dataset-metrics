#!/usr/bin/env python3
"""
Base class for model handling, providing shared functionality for training and chat interfaces.
"""

import logging
import yaml
from pathlib import Path
from typing import Any

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

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

    def load_model_and_tokenizer(self) -> tuple[Any, Any]:
        """Load model and tokenizer with appropriate configuration.

        :return: Tuple of (model, tokenizer)
        :rtype: tuple[Any, Any]
        """
        self.log.info(f"Loading model: {self.config['model_name']}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config["model_name"],
            max_seq_length=self.config["max_seq_length"],
            dtype=self.config["dtype"],
            load_in_4bit=self.config["load_in_4bit"],
        )

        tokenizer = get_chat_template(
            tokenizer,
            chat_template=self.model_settings["chat_template"],
        )

        return model, tokenizer
