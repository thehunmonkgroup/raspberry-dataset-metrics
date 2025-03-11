#!/usr/bin/env python3
"""
Universal chat interface for interacting with fine-tuned language models.
Loads configuration from YAML files and abstracts model-specific logic.
"""

import argparse
import logging
import re
from re import Pattern
import readline
import sys
import yaml
from pathlib import Path
from typing import Any

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from peft.peft_model import PeftModel

import torch
from transformers import TextStreamer

from raspberry_dataset_metrics import constants
from raspberry_dataset_metrics import util


class CaptureTextStreamer(TextStreamer):
    """Custom text streamer that captures and displays generated text."""

    def __init__(self, tokenizer: Any, skip_prompt: bool = False, eos_token: str = "") -> None:
        """Initialize the streamer.

        :param tokenizer: The tokenizer to use for decoding
        :type tokenizer: Any
        :param skip_prompt: Whether to skip the prompt in the output
        :type skip_prompt: bool
        :param eos_token: End of sequence token to handle in display
        :type eos_token: str
        """
        super().__init__(tokenizer, skip_prompt)
        self.captured_text: list[str] = []
        self.eos_token: str = eos_token

    def put(self, value: Any) -> None:  #pyright: ignore[reportImplicitOverride]
        """Process and display a token.

        :param value: Token or tensor to process
        :type value: Any
        """
        if torch.is_tensor(value):
            value = value.cpu()
            text = self.tokenizer.decode(value[0] if value.dim() > 1 else value)  # pyright: ignore[reportAttributeAccessIssue]
            display_text = text.replace(self.eos_token, "")  # Remove tag for display only
            if display_text.strip():  # Only display if there's content
                super().put(value)  # Pass the original tensor to super().put()
            if text.strip():  # Always append original text (with tag) to captured_text
                self.captured_text.append(text)
        else:
            super().put(value)


class Chat:
    """Main class for interacting with fine-tuned language models using configuration from YAML files."""

    def __init__(self, config_path: Path, checkpoint: str | None = None, debug: bool = False) -> None:
        """Initialize chat interface with configuration from YAML.

        :param config_path: Path to the YAML configuration file
        :type config_path: Path
        :param checkpoint: Checkpoint directory
        :type checkpoint: str | None
        :param debug: Enable debug logging
        :type debug: bool
        """
        self.config_path: Path = config_path
        self.checkpoint: str | None = checkpoint
        self.config: dict[str, Any] = self._load_config(self.config_path)
        self.logger: logging.Logger = self._setup_logging(debug)
        self.logger.info(f"Initializing chat with configuration: {self.config_path}")
        if self.checkpoint:
            self.logger.info(f"Using checkpoint: {self.checkpoint}")
        self.logger.debug(f"Configuration: {self.config}")
        self.model_settings: dict[str, Any] = self._get_model_family_settings()
        self.messages: list[dict[str, str]] = []
        self.response_extraction_pattern: Pattern[str] = re.compile(self.model_settings["response_extraction_pattern"], re.DOTALL)

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
                "model_name": "",
                "model_family": "",
                "max_seq_length": constants.MAX_SEQ_LENGTH,
                "dtype": constants.DTYPE,
                "load_in_4bit": constants.LOAD_IN_4BIT,
                "system_message": constants.SYSTEM_MESSAGE,
                "temperature": constants.TEMPERATURE,
                "min_p": constants.MIN_P,
                "max_new_tokens": constants.MAX_NEW_TOKENS,
            }
            merged_config.update(config)
            return merged_config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

    def _setup_logging(self, debug: bool) -> logging.Logger:
        """Configure logging for the chat interface.

        :param debug: Enable debug logging
        :type debug: bool
        :return: Configured logger
        :rtype: logging.Logger
        """
        logger = logging.getLogger("chat")
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _get_model_family_settings(self) -> dict[str, Any]:
        """Get model-specific settings based on the model family.

        :return: Dictionary of model-specific settings
        :rtype: dict[str, Any]
        :raises ValueError: If model family is not supported
        """
        model_family = self.config.get("model_family")
        if not model_family or model_family not in constants.MODEL_FAMILIES:
            raise ValueError("Unsupported or unspecified model family. Please specify 'model_family' in config.")
        settings = constants.MODEL_FAMILIES[model_family].copy()
        return settings

    def init_messages(self) -> list[dict[str, str]]:
        """Initialize conversation with system message.

        :return: List of message dictionaries
        :rtype: List[Dict[str, str]]
        """
        system_message = self.config.get("system_message", constants.SYSTEM_MESSAGE)
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        return messages

    def load_model(self) -> tuple[Any, Any]:
        """Load model and tokenizer with appropriate configuration.

        :return: Tuple of (model, tokenizer)
        :rtype: tuple[Any, Any]
        """
        self.logger.info(f"Loading model: {self.config['model_name']}")
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
        if self.checkpoint:
            output_dir = self.config.get("output_dir", f"outputs/{util.get_config_base_name(self.config_path)}")
            checkpoint_path = Path(output_dir) / self.checkpoint
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Error: Checkpoint directory not found: {checkpoint_path}")
            if not (checkpoint_path / "adapter_model.safetensors").exists():
                raise FileNotFoundError(f"Error: No adapter model found in {checkpoint_path}")
            self.logger.info(f"Loading adapter weights from checkpoint {checkpoint_path}")
            model = PeftModel.from_pretrained(model, checkpoint_path)
        model = FastLanguageModel.for_inference(model)
        if self.checkpoint:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
        return model, tokenizer

    def process_response(self, response: str) -> str | None:
        """Extract assistant's response from the model output.

        :param response: Raw model output
        :type response: str
        :return: Extracted assistant response or None if extraction failed
        :rtype: Optional[str]
        """
        match = self.response_extraction_pattern.search(response)
        if match:
            return match.group(1).strip()
        else:
            self.logger.warning(f"Could not extract assistant response from: {response}")
            return None

    def _setup_readline(self) -> None:
        """Configure readline with in-memory history for the current session.
        """
        readline.set_history_length(1000)
        self.logger.debug("Readline configured with in-memory history")

    def run(self) -> None:
        """Run the chat interface.

        :raises Exception: If model loading or inference fails
        """
        self.logger.info("Starting chat interface")
        model, tokenizer = self.load_model()
        self.messages = self.init_messages()
        self._setup_readline()
        print("\nChat interface started (Press Ctrl+C to exit)")
        print("Special commands: /exit to quit, /new to start a new conversation")
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.strip() == "/exit":
                    print("Chat interface exited.")
                    return
                elif user_input.strip() == "/new":
                    print("Starting new conversation.")
                    self.messages = self.init_messages()
                    continue
                print("\nAssistant:\n")
                self.messages.append({"role": "user", "content": user_input})
                inputs = tokenizer.apply_chat_template(
                    self.messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to("cuda")
                text_streamer = CaptureTextStreamer(
                    tokenizer,
                    skip_prompt=True,
                    eos_token=self.model_settings["eos_token"]
                )
                eos_token_id = tokenizer.convert_tokens_to_ids(self.model_settings["eos_token"])
                _ = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    streamer=text_streamer,
                    max_new_tokens=self.config["max_new_tokens"],
                    eos_token_id=eos_token_id,
                    use_cache=True,
                    temperature=self.config["temperature"],
                    min_p=self.config["min_p"],
                )
                response = "".join(text_streamer.captured_text)
                assistant_response = self.process_response(response)
                if assistant_response:
                    self.messages.append({"role": "assistant", "content": assistant_response})
            except KeyboardInterrupt:
                print('\nExiting chat interface')
                sys.exit(0)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    :return: Parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Chat interface for fine-tuned language models"
    )
    parser.add_argument(
        "config_file",
        help="Path to the YAML config file"
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint directory (e.g., 'checkpoint-60')"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for the chat interface.

    :return: Exit code
    :rtype: int
    """
    try:
        args = parse_args()
        config_path = Path(args.config_file)
        chat = Chat(config_path, args.checkpoint, args.debug)
        chat.run()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
