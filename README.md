# Raspberry Dataset Metrics

Code for testing the quality of datasets generated for the [Raspberry](https://github.com/daveshap/Raspberry) project.

Supports fine-tuning and interacting with large language models using Unsloth.

## Features

- Universal training script with configurable hyperparameters via YAML
- Interactive chat interface for testing fine-tuned models
- Support for multiple model families with appropriate templates
- Modular design with model-specific settings abstracted away

## Installation

```bash
# Install from source
pip install .

# Install with development dependencies
pip install ".[dev]"
```

## Usage

### Training

Fine-tune models with custom datasets:

```bash
raspberry-train configs/llama-3.1-8b.yaml dataset.json
```

### Chat Interface

Interact with models (base or fine-tuned):

```bash
# Use base model
raspberry-chat configs/phi-4.yaml

# Use fine-tuned checkpoint
raspberry-chat configs/phi-4.yaml --checkpoint checkpoint-60
```

## Configuration

Configuration is done via YAML files in the `configs` directory. Each config specifies:

- `model_name`: Hugging Face model ID
- `model_family`: Model family for template handling (e.g., "llama-3.1", "phi-4")
- Optional hyperparameters that override defaults from `constants.py`

### Example Configuration

```yaml
model_name: "unsloth/phi-4-unsloth-bnb-4bit"
model_family: "phi-4"
warmup_steps: 5
learning_rate: 2e-4
```

## Supported Models

- Llama 3.1/3.2 series
- Phi-4
- Qwen 2.5
- Easily extensible to other models
