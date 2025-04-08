# Toxicity Classifier

When training large language models (LLMs), it's critical to ensure that your dataset does not contain toxic content. This repository provides scripts for labeling data, fine-tuning toxicity classifiers, and performing distributed inference to detect toxic content.

## Repository Structure

```
toxicity-classifier/
├── .github/
├── requirements/
│   ├── base.txt
│   └── ...
├── src/
│   ├── finetune_bert_classifier.py
│   ├── finetune_bert_regresor.py
│   ├── finetune_fasttext_classifier.py
│   ├── gcp_data_labeling.py
│   ├── inference_bert_regresor.py
│   └── inference_fasttext_classifier.py
├── .flake8
├── .gitignore
├── pre-commit-config.yaml
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

## Usage

This repository has three main modules:

### 1. GCP-based Data Labeling

The script `gcp_data_labeling.py` labels a subsample of your dataset using the Google Cloud Natural Language API. Ensure the API is enabled and set up a service account.

**Example:**

```bash
python src/gcp_data_labeling.py --service_account_file <path_to_service_account.json> \
                                --dataset_path <path_to_dataset> \
                                --output_file <output_file_path> \
                                --text_column <text_column_name>
```

### 2. Fine Tuning

After labeling your data, fine-tune models using these scripts:

#### BERT Classifier

```bash
python src/finetune_bert_classifier.py --json_path <path_to_labeled_data.json> \
                                       --output_dir <output_directory> \
                                       --logging_dir <logging_directory> \
                                       --threshold <confidence_threshold> \
                                       --model_name <hf_model_name> \
                                       --undersample
```

#### BERT Regressor

For continuous toxicity scores:

```bash
python src/finetune_bert_regresor.py --json_path <path_to_labeled_data.json> \
                                     --output_dir <output_directory> \
                                     --logging_dir <logging_directory> \
                                     --label_id <label_index> \
                                     --threshold <score_threshold> \
                                     --model_name <hf_model_name> \
                                     --undersample
```

#### fastText Classifier

Fine-tune a fastText classifier:

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

- The flags `--undersample` or `--oversample` activate dataset balancing methods.

### 3. Inference

Perform distributed inference on your entire dataset:

#### fastText Classifier Inference

```bash
python src/inference_fasttext_classifier.py --model_path <path_to_trained_model.bin> \
                                            --dataset_path <path_to_dataset> \
                                            --output_path <output_file_or_directory> \
                                            --text_column <text_column_name> \
                                            --batch_size <batch_size> \
                                            --chunk_size <examples_per_chunk>
```

#### BERT Regressor Inference

```bash
python src/inference_bert_regresor.py --model_path <path_to_trained_model> \
                                      --dataset_path <path_to_dataset> \
                                      --output_path <output_directory> \
                                      --text_column <text_column_name> \
                                      --batch_size <batch_size> \
                                      --procs_per_gpu <procs_per_gpu>
```

## Additional Notes

- **GPU/CPU Usage:**
  - BERT-based scripts utilize GPUs via `CUDA_VISIBLE_DEVICES`.
  - fastText scripts typically run on CPUs but can support GPUs experimentally.

- **Hyperparameters:**
  - Print fastText model hyperparameters for verification using provided functions.

- **Chunked Processing:**
  - Inference scripts handle large datasets efficiently by processing data in chunks.

## License

This project is licensed under the [MIT License](./LICENSE).

