#!/usr/bin/env python3
"""
Universal chat interface for interacting with fine-tuned language models.
Loads configuration from YAML files and abstracts model-specific logic.
"""

import argparse
import re
from re import Pattern
import sys
from pathlib import Path
from typing import Any
from collections.abc import Iterator

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.shortcuts import clear

import torch
from transformers import TextStreamer

from raspberry_dataset_metrics import constants
from raspberry_dataset_metrics import util
from raspberry_dataset_metrics.base_model import BaseModelHandler


class ChatCommandCompleter(Completer):
    """Completer for chat commands."""

    def __init__(self) -> None:
        """Initialize the command completer with available commands."""
        self.commands: list[tuple[str, str]] = [
            ("/exit", "Quit the chat"),
            ("/new", "Start a new conversation"),
            ("/help", "Show available commands"),
            ("/clear", "Clear the screen"),
            ("/temp", "View or adjust temperature (e.g., /temp 0.7)"),
            ("/system", "View or modify system message (e.g., /system New message)"),
        ]

    def get_completions(  # pyright: ignore[reportImplicitOverride]
        self, document: Any, complete_event: Any
    ) -> Iterator[Completion]:
        """Get command completions based on current text.

        :param document: Document containing current input
        :type document: Any
        :param complete_event: Completion event info
        :type complete_event: Any
        :return: Iterator of completions
        :rtype: Iterator[Completion]
        """
        text = document.text_before_cursor.lstrip()

        # Only suggest commands at the start of input
        if text.startswith("/"):
            for command, help_text in self.commands:
                if command.startswith(text):
                    yield Completion(
                        command,
                        start_position=-len(text),
                        display=command,
                        display_meta=help_text
                    )


class RichTextStreamer(TextStreamer):
    """Custom text streamer that formats and displays model output with Rich styling."""

    def __init__(
        self, tokenizer: Any, skip_prompt: bool = True, eos_token: str = "", console: Console | None = None
    ) -> None:
        """Initialize the streamer with Rich console integration.

        :param tokenizer: The tokenizer to use for decoding
        :type tokenizer: Any
        :param skip_prompt: Whether to skip the prompt in the output
        :type skip_prompt: bool
        :param eos_token: End of sequence token to handle in display
        :type eos_token: str
        :param console: Rich console instance for formatted output
        :type console: Console
        """
        # Always skip prompt - we only want to display the assistant's response
        super().__init__(tokenizer, skip_prompt=True)
        self.console: Console = console or Console()
        self.captured_text: list[str] = []
        self.eos_token: str = eos_token
        self.buffer: str = ""
        self.in_reasoning_tag: bool = False
        self.in_output_tag: bool = False

    def put(self, value: Any) -> None:
        """Process, format, and display a token with Rich styling.

        :param value: Token or tensor to process
        :type value: Any
        """
        if not torch.is_tensor(value):
            return super().put(value)

        value = value.cpu()
        text = self.tokenizer.decode(
            value[0] if value.dim() > 1 else value
        )

        # Store the raw text for capturing full output
        if text.strip():
            self.captured_text.append(text)

        # Let the parent class handle prompt skipping
        # This is the key change - delegate to parent for display decisions
        super().put(value)

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        """Process finalized text chunks after prompt skipping.

        :param text: Text chunk to process and display
        :type text: str
        :param stream_end: Flag indicating if this is the final chunk
        :type stream_end: bool
        """
        # Remove EOS token for display
        display_text = text.replace(self.eos_token, "")

        # Skip empty tokens
        if not display_text.strip():
            return

        # Add to our buffer and process tags
        self.buffer += display_text
        self._process_buffer(final=stream_end)

    def _process_buffer(self, final: bool = False) -> None:
        """Process the buffer to detect and format XML tags.

        :param final: Flag indicating if this is the final processing call
        :type final: bool
        """
        # Check for opening reasoning tag
        if "<reasoning>" in self.buffer and not self.in_reasoning_tag:
            parts = self.buffer.split("<reasoning>", 1)
            # Print text before tag
            if parts[0]:
                self.console.print(Text(parts[0]), end="")
            # Use Text object to avoid markup parsing
            self.console.print(Text("\nREASONING\n", style="reasoning"), end="")
            self.buffer = parts[1]
            self.in_reasoning_tag = True

        # Check for closing reasoning tag
        if "</reasoning>" in self.buffer and self.in_reasoning_tag:
            parts = self.buffer.split("</reasoning>", 1)
            # Print reasoning content with Text object
            self.console.print(Text(parts[0], style="reasoning"), end="")
            self.console.print(Text("\n/REASONING\n", style="reasoning"), end="")
            self.buffer = parts[1]
            self.in_reasoning_tag = False

        # Check for opening output tag
        if "<output>" in self.buffer and not self.in_output_tag:
            parts = self.buffer.split("<output>", 1)
            # Print text before tag
            if parts[0]:
                self.console.print(Text(parts[0]), end="")
            self.console.print(Text("\nOUTPUT\n", style="output"), end="")
            self.buffer = parts[1]
            self.in_output_tag = True

        # Check for closing output tag
        if "</output>" in self.buffer and self.in_output_tag:
            parts = self.buffer.split("</output>", 1)
            # Print output content with Text object
            self.console.print(Text(parts[0], style="output"), end="")
            self.console.print(Text("\n/OUTPUT\n", style="output"), end="")
            self.buffer = parts[1]
            self.in_output_tag = False

        # Print any remaining text with appropriate style if no pending tags
        # Or if this is the final call, print whatever is left in the buffer
        if final or not ("<" in self.buffer and ">" in self.buffer):
            style = "reasoning" if self.in_reasoning_tag else "output" if self.in_output_tag else ""
            # Use Text object instead of styled string
            if self.buffer:
                self.console.print(Text(self.buffer, style=style), end="")
                self.buffer = ""


class Chat(BaseModelHandler):
    """Main class for interacting with fine-tuned language models using configuration from YAML files."""

    def __init__(
        self, config_path: Path, fine_tune: bool = False, debug: bool = False
    ) -> None:
        """Initialize chat interface with configuration from YAML.

        :param config_path: Path to the YAML configuration file
        :type config_path: Path
        :param fine_tune: Load fine-tuned model
        :type fine_tune: bool
        :param debug: Enable debug logging
        :type debug: bool
        """
        super().__init__(config_path, debug)
        self.fine_tune: bool = fine_tune
        if self.fine_tune:
            self.log.info("Using fine-tuned model")
        self.messages: list[dict[str, str]] = []
        self.response_extraction_pattern: Pattern[str] = re.compile(
            self.model_settings["response_extraction_pattern"], re.DOTALL
        )

        # Setup Rich console for formatted output
        custom_theme = Theme({
            "user": "bold cyan",
            "assistant": "bold green",
            "system": "bold yellow",
            "info": "bold blue",
            "warning": "bold red",
            "reasoning": "italic yellow",
            "output": "bold white",
        })
        self.console: Console = Console(theme=custom_theme)
        self.command_completer: ChatCommandCompleter = ChatCommandCompleter()

    def init_messages(self) -> list[dict[str, str]]:
        """Initialize conversation with system message.

        :return: List of message dictionaries
        :rtype: list[dict[str, str]]
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
        model, tokenizer = self.load_model_and_tokenizer()

        if self.fine_tune:
            self.load_peft_model(model, tokenizer)
        model.eval()
        torch.set_grad_enabled(False)
        try:
            if not self.config.get("load_in_4bit", False):
                self.log.info("Compiling model with torch.compile().")
                model = torch.compile(model)
        except Exception:
            self.log.warning("Unable to compile model with torch.compile().")
            pass
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
            self.log.warning(
                f"Could not extract assistant response from: {response}"
            )
            return None

    def _show_help(self) -> None:
        """Display help information about available commands."""
        command_text = "Available Commands:\n\n"
        for command, help_text in self.command_completer.commands:
            command_text += f"{command} - {help_text}\n"

        command_text += "\nMulti-line Input:\n"
        command_text += "End a line with \\ to continue to the next line\n"

        self.console.print(Panel(
            Text(command_text, justify="left"),
            title="Help",
            border_style="blue"
        ))

    def _adjust_temperature(self, args: str | None = None) -> None:
        """Display or adjust model temperature setting.

        :param args: Optional temperature value
        :type args: str | None
        """
        if not args:
            # Display current temperature
            self.console.print(f"[info]Current temperature: {self.config['temperature']}[/info]")
            return

        try:
            temp = float(args.strip())
            if temp <= 0:
                self.console.print("[warning]Temperature must be a positive number[/warning]")
                return

            # Update temperature in config
            self.config['temperature'] = temp
            self.console.print(f"[info]Temperature updated to: {temp}[/info]")
        except ValueError:
            self.console.print("[warning]Invalid temperature value. Please provide a positive number.[/warning]")

    def _manage_system_message(self, args: str | None = None) -> None:
        """Display or modify the system message.

        :param args: New system message text
        :type args: str | None
        """
        # Find system message in conversation history
        system_idx = None
        for idx, msg in enumerate(self.messages):
            if msg["role"] == "system":
                system_idx = idx
                break

        if not args:
            # Display current system message
            system_message = self.messages[system_idx]["content"] if system_idx is not None else "No system message set"
            self.console.print(Panel(
                Text(system_message),
                title="Current System Message",
                border_style="yellow"
            ))
            return

        # Update system message
        new_system_message = args.strip()
        if system_idx is not None:
            # Update existing system message
            self.messages[system_idx]["content"] = new_system_message
        else:
            # Add new system message at the beginning
            self.messages.insert(0, {"role": "system", "content": new_system_message})

        # Update in config too for future new conversations
        self.config["system_message"] = new_system_message
        self.console.print("[info]System message updated successfully[/info]")


    def _setup_prompt_session(self) -> PromptSession[Any]:
        """Configure prompt toolkit session with history and key bindings.

        :return: Configured prompt session
        :rtype: PromptSession
        """
        # Create key bindings with multi-line support
        kb = KeyBindings()

        @kb.add('enter')
        def _(event: Any) -> None:
            """Submit on Enter unless Shift is pressed."""
            if event.current_buffer.document.text.strip().endswith('\\'):
                # Get current text and cursor position
                text = event.current_buffer.text
                position = event.current_buffer.cursor_position

                # Find where the backslash is relative to the cursor
                line_end_pos = text.rfind('\\', 0, position)
                if line_end_pos >= 0:
                    # Remove backslash and add newline
                    new_text = text[:line_end_pos] + '\n' + text[line_end_pos+1:]
                    event.current_buffer.text = new_text

                    # Move cursor to after the newline
                    event.current_buffer.cursor_position = line_end_pos + 1
                else:
                    # Fallback: just add a newline at cursor
                    event.current_buffer.insert_text('\n')
            else:
                event.current_buffer.validate_and_handle()

        # Create style
        style = Style.from_dict({
            'prompt': 'cyan bold',
            'continuation': 'gray italic',
        })

        # Create and return the session
        session = PromptSession(
            history=InMemoryHistory(),
            completer=self.command_completer,
            complete_while_typing=True,
            key_bindings=kb,
            style=style,
            multiline=True,
            mouse_support=True,
            wrap_lines=True,
        )

        self.log.debug("Prompt toolkit session configured")
        return session

    def run(self) -> None:
        """Run the chat interface.

        :raises Exception: If model loading or inference fails
        """
        self.log.info("Starting chat interface")
        model, tokenizer = self.load_model()
        do_sample = model.generation_config.do_sample
        self.log.info(f"Sampling enabled: {do_sample}")
        self.messages = self.init_messages()
        prompt_session = self._setup_prompt_session()
        label = f" {self.config['label']}" if "label" in self.config else ""

        # Display welcome header with rich formatting
        self.console.print(Panel(
            Text("Welcome to the AI Chat Interface", justify="center"),
            border_style="blue",
            title=f"Raspberry Dataset Chat{label}",
            width=self.console.width,
            expand=True,
            padding=(1, 2),
            title_align="center"
        ))
        self.console.print("[info]Type /help to list commands[/info]")

        while True:
            try:
                # Get input using prompt_toolkit with HTML formatting
                user_input = prompt_session.prompt([
                    ('class:prompt', '\nYou: '),
                ])

                # Remove trailing backslash used for multi-line input
                user_input = user_input.rstrip('\\').strip()

                # Handle commands
                user_input = user_input.strip()
                if user_input == "/exit":
                    self.console.print("[info]Chat interface exited.[/info]")
                    return
                elif user_input == "/new":
                    self.console.print("[info]Starting new conversation.[/info]")
                    self.messages = self.init_messages()
                    continue
                elif user_input == "/clear":
                    clear()
                    continue
                elif user_input == "/help":
                    self._show_help()
                    continue
                elif user_input.startswith("/temp"):
                    args = user_input[5:].strip() if len(user_input) > 5 else None
                    self._adjust_temperature(args)
                    continue
                elif user_input.startswith("/system"):
                    args = user_input[7:].strip() if len(user_input) > 7 else None
                    self._manage_system_message(args)
                    continue
                elif not user_input:
                    continue

                # Add the user message to history (no need to display again)
                self.messages.append({"role": "user", "content": user_input})

                # Display assistant indicator with appropriate styling
                self.console.print()  # Add spacing
                self.console.print("[assistant]Assistant:[/assistant]")
                self.console.print()  # Add spacing

                # Prepare inputs for the model
                inputs = tokenizer.apply_chat_template(
                    self.messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to("cuda")

                # Create streamer that will handle assistant's response
                text_streamer = RichTextStreamer(
                    tokenizer,
                    eos_token=tokenizer.eos_token,
                    console=self.console
                )
                generation_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "streamer": text_streamer,
                    "max_new_tokens": self.config["max_new_tokens"],
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                    "use_cache": True,
                    "remove_invalid_values": True,    # adds InfNanRemoveLogitsProcessor
                    "renormalize_logits": True,       # fixes sums that drift from 1
                    "do_sample": do_sample,
                }
                if do_sample:
                    generation_kwargs.update({
                        "temperature": self.config["temperature"],
                        "min_p": self.config["min_p"],
                    })
                _ = model.generate(**generation_kwargs)
                response = "".join(text_streamer.captured_text)
                assistant_response = self.process_response(response)
                if assistant_response:
                    # No need for additional formatting since it's already displayed during streaming
                    self.messages.append(
                        {"role": "assistant", "content": assistant_response}
                    )
            except KeyboardInterrupt:
                self.console.print("\n[warning]Exiting chat interface[/warning]")
                sys.exit(0)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    :return: Parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Chat interface for fine-tuned language models"
    )
    parser.add_argument("config_file", help="Path to the YAML config file")
    parser.add_argument(
        "--fine-tune", action="store_true", help="Load fine-tuned model"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    """Main entry point for the chat interface.

    :return: Exit code
    :rtype: int
    """
    try:
        args = parse_args()
        config_path = Path(args.config_file)
        chat = Chat(config_path, args.fine_tune, args.debug)
        chat.run()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
