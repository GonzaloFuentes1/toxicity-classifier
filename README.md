# Toxicity Classifier

When training large language models (LLMs), it's critical to ensure that your dataset does not contain toxic content. This repository provides comprehensive scripts for labeling data, fine-tuning toxicity classifiers, and performing distributed inference to detect toxic content across multiple categories.

## Repository Structure

```
toxicity-classifier/
├── .github/
├── requirements/
│   ├── base.txt
│   └── dev.txt
├── src/
│   ├── finetune_bert_classifier.py
│   ├── finetune_bert_multiple_labels.py
│   ├── finetune_fasttext_classifier.py
│   ├── gcp_data_labeling.py
│   ├── inference_bert_multilabel.py
│   ├── inference_fasttext_classifier.py
│   ├── moderate_llm.py
│   ├── prompts.py
│   └── split_and_tokenize.py
├── .flake8
├── .gitignore
├── .pre-commit-config.yaml
└── README.md
```

## Installation

1. **Create a Python environment:**

```bash
python -m venv venv
source venv/bin/activate
```

On Windows, activate using:

```bash
venv\Scripts\activate
```

2. **Install dependencies:**

```bash
pip install -r requirements/base.txt
```

For development:

```bash
pip install -r requirements/dev.txt
pip install pre-commit
pre-commit install
```

## Usage

This repository has four primary modules:

### 1. Data Labeling

#### GCP-based Data Labeling

The script `gcp_data_labeling.py` labels a subsample of your dataset using the Google Cloud Natural Language API, which provides toxicity and sentiment scores. You must ensure the API is enabled and set up a service account with appropriate permissions.

**Example:**

```bash
python src/gcp_data_labeling.py --service_account_file <path_to_service_account.json> \
                                --dataset_path <path_to_dataset> \
                                --output_file <output_file_path> \
                                --text_column <text_column_name>
```

#### LLM-based Data Labeling

The script `moderate_llm.py` uses large language models to label toxic content with detailed categories (Toxic, Insult, Profanity, Sexual, Violent). It supports multiple models for consensus-based scoring and uses vLLM for efficient batched inference.

**Example:**

```bash
python src/moderate_llm.py --model_names TheBloke/Mistral-7B-Instruct-v0.2 meta-llama/Llama-2-7b-chat-hf \
                           --download_dir <model_download_directory> \
                           --dataset_path <path_to_dataset> \
                           --output_path <output_results_path> \
                           --text_column <text_column_name> \
                           --batch_size 32 \
                           --cuda_devices "0,1"
```

### 2. Data Preprocessing

The `split_and_tokenize.py` script splits texts into paragraphs and tokenizes them for training or inference, handling the preparation of data for both BERT and fastText models.

**Example:**

```bash
python src/split_and_tokenize.py --dataset_path <path_to_dataset> \
                                --text_column <text_column_name> \
                                --expanded_path <expanded_dataset_path> \
                                --tok_path <tokenized_dataset_path> \
                                --model_path <hf_model_path> \
                                --max_length 512
```

### 3. Fine Tuning

After labeling your data, fine-tune models using these specialized scripts:

#### BERT Binary Classifier

Train a binary classifier for toxic content detection:

```bash
python src/finetune_bert_classifier.py --json_path <path_to_labeled_data.json> \
                                       --output_dir <output_directory> \
                                       --logging_dir <logging_directory> \
                                       --threshold <confidence_threshold> \
                                       --model_name <hf_model_name> \
                                       --undersample
```

#### BERT Multi-label Classifier

For classification across multiple toxicity categories:

```bash
python src/finetune_bert_multiple_labels.py --json_path <path_to_labeled_data.json> \
                                           --output_dir <output_directory> \
                                           --logging_dir <logging_directory> \
                                           --label_ids 0 1 2 3 4 \
                                           --threshold <score_threshold> \
                                           --model_name <hf_model_name> \
                                           --balance_method oversample
```

#### fastText Classifier

Fine-tune a lightweight, efficient fastText classifier:

```bash
python src/finetune_fasttext_classifier.py --json_path <path_to_labeled_data.json> \
                                           --output_model <path_to_output_model.bin> \
                                           --threshold <confidence_threshold> \
                                           --word_ngrams 2 \
                                           --label "Toxic" \
                                           --epoch 25 \
                                           --lr 1.0 \
                                           --autotuneDuration 600 \
                                           --oversample
```

- Use `--undersample` to reduce majority class examples or `--oversample` to duplicate minority class examples for handling imbalanced datasets.
- The fastText model supports auto-tuning via the `--autotuneDuration` parameter.

### 4. Inference

Perform distributed inference on your entire dataset:

#### fastText Classifier Inference

For efficient CPU-based inference with paragraph-level toxicity detection:

```bash
python src/inference_fasttext_classifier.py --model_path <path_to_trained_model.bin> \
                                            --dataset_path <path_to_dataset> \
                                            --output_path <output_directory> \
                                            --text_column <text_column_name> \
                                            --batch_size 64 \
                                            --chunk_size 10000
```

#### BERT Multi-label Inference

For distributed GPU-based inference with multiple toxicity categories:

```bash
python src/inference_bert_multilabel.py --tok_path <tokenized_dataset_path> \
                                       --orig_path <original_dataset_path> \
                                       --expanded_path <expanded_dataset_path> \
                                       --model_path <model_checkpoint_path> \
                                       --batch_size 128 \
                                       --threshold 0.5 \
                                       --output_path <output_prefix> \
                                       --shard_id 0 \
                                       --num_shards 4
```

- The `--shard_id` and `--num_shards` parameters enable distributed processing across multiple GPUs or machines.

## Additional Notes

- **GPU/CPU Utilization:**

  - BERT-based scripts utilize GPUs through `CUDA_VISIBLE_DEVICES` or the `--cuda_devices` parameter.
  - The fastText scripts primarily run on CPUs but are optimized for multi-core processing.
  - The LLM-based moderation uses vLLM for efficient tensor-parallel inference across multiple GPUs.

- **Model Hyperparameters:**

  - The fastText scripts include functions to print model hyperparameters for verification and debugging.
  - BERT models support various training configurations through the Transformers Trainer API.

- **Efficient Data Processing:**

  - All inference scripts handle large datasets efficiently by processing data in chunks.
  - The paragraph-based approach enables more precise toxicity detection than document-level classification.
  - Multi-shard processing allows distributed inference across multiple machines or GPUs.

- **Code Formatting:**
  - The repository enforces consistent code formatting with a maximum line length of 88 characters.
  - Pre-commit hooks for black, isort, and flake8 ensure code quality and consistency.

## License

This project is licensed under the [MIT License](./LICENSE).
