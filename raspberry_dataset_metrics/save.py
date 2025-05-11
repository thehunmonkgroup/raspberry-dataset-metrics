#!/usr/bin/env python3
"""
Universal save script for fine-tuned language models.
"""

import datetime
import argparse
from pathlib import Path
import sys

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from raspberry_dataset_metrics import util
from raspberry_dataset_metrics.base_model import BaseModelHandler


class Saver(BaseModelHandler):
    """Main class for saving language models using configuration from YAML files."""

    def __init__(
        self, config_path: Path, repo_tag: str | None, debug: bool = False
    ) -> None:
        """Initialize saver with configuration from YAML.

        :param config_path: Path to the YAML configuration file
        :type config_path: Path
        :param repo_tag: Optional tag for the Hugging Face Hub repository
        :type repo_tag: str | None
        :param debug: Enable debug logging
        :type debug: bool
        """
        super().__init__(config_path, debug)
        self.log.info(f"Initializing saver for: {self.label}")
        self.repo_name: str = self.build_repo_name(repo_tag)

    def build_repo_name(self, repo_tag: str | None) -> str:
        repo_tag = repo_tag or datetime.datetime.now().strftime('%Y-%m-%dT%H:%M')
        return f"{util.get_config_base_name(self.config_path)}-{repo_tag}"

    def save(self) -> None:
        """Save the model and tokenizer.
        """
        base_model, tokenizer = self.load_model_and_tokenizer()
        peft_model = self.load_peft_model(base_model, tokenizer)
        self.log.info(f"Pushing tokenizer to Hugging Face Hub: {self.repo_name}")
        tokenizer.push_to_hub(self.repo_name)
        self.log.info(f"Pushing LoRA adapter to Hugging Face Hub: {self.repo_name}")
        peft_model.push_to_hub(self.repo_name)
        self.log.info("✅ Push complete!")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    :return: Parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Train language models with configurations from YAML files"
    )
    parser.add_argument("config_file", help="Path to the YAML config file")
    parser.add_argument("--repo-tag", help="Additional tag for the Hugging Face Hub repository, defaults to current date and time")
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
        saver = Saver(config_path, args.repo_tag, args.debug)
        saver.save()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
