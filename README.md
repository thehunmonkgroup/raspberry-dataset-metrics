# Raspberry Dataset Metrics

Code for testing the quality of datasets generated for the [Raspberry](https://github.com/daveshap/Raspberry) project.

Supports fine-tuning and interacting with large language models using standard HuggingFace libraries.


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


## Setup

A [HuggingFace API token](https://huggingface.co/settings/tokens) is required, as some models are gated.

For gated models, you will need to complete the application process to access it.

The scripts will look for a `HUGGINGFACEHUB_API_TOKEN` environment variable to automatically load the token:

```bash
export HUGGINGFACEHUB_API_TOKEN="hf_yourhuggingfacetoken"
```

If not set, you will be prompted for the token.


## Usage

### Training

Fine-tune models with custom datasets:

```bash
raspberry-train configs/llama-3.1-8b.yaml dataset.jsonl
```

By default, the PEFT adapter and tokenizer files will be saved to an `outputs/[config_basename]` directory, e.g. `outputs/llama-3.1-8b`

### Chat Interface

Interact with models (base or fine-tuned):

```bash
# Use base model
raspberry-chat configs/phi-4.yaml

# Use the fine-tuned model
raspberry-chat configs/phi-4.yaml --fine-tune
```

### Saving models to Hugging Face

Models can be easily saved to the associated HuggingFace user:

```bash
# Save using the default tag of YYYY_MM_DD_HH_MM
raspberry-save configs/phi-4.yaml

# Save using a custom tag
raspberry-save configs/phi-4.yaml --repo-tag 001
```


## Configuration

Configuration is done via YAML files in the `configs` directory. Each config specifies:

- `label`: Optional human-readable label
- `model_name`: Hugging Face model ID
- `model_family`: Model family for template handling (e.g., "llama-3.1", "phi-4")
- Optional hyperparameters that override defaults from `constants.py` (see [utils.py](raspberry_dataset_metrics/utils.py) for the all parameter labels)

### Example Configuration

```yaml
model_name: "meta-llama/Llama-3.1-8B-Instruct"
model_family: "llama-3.1"
warmup_steps: 5
learning_rate: 2e-4
```


## Supported Models

- Llama 3.1/3.2 series
- Mistral 7B
- Phi-4
- Qwen 2.5
- Easily extensible to other models
