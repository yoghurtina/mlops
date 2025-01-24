# MLOps: Large Language Model (LLM) for the Greek Language

## Project Description

### Overall Goal of the Project
The primary goal of this project is to fine-tune and evaluate a 
lightweight Large Language Model (LLM) for general-purpose text generation. 
By focusing on an accessible model like GPT-2, this project aims to enable 
efficient training and deployment on limited hardware, 
making it suitable for small-scale experiments and projects.

We aim to follow MLOps best practices to make development efficient and reproducible,
emphasizing ease of use and simplicity in deployment.

## Framework and Integration

We use the following tools and frameworks for the development and deployment of the project:

- **PyTorch Lightning**: for modular and scalable training workflows.
- **Transformers (Hugging Face)**: use of pre-trained models and utilities for fine-tuning and text generation.
- **Datasets (Hugging Face)**: for easy access to standardized datasets for training and evaluation.
- **Docker**: for consistent development environment through containerization, simplifying deployment and collaboration.
- **Hydra**: for flexible configuration management, enabling easy customization of hyperparameters, datasets, and training settings.
- **CUDA/cuDNN**: to accelerate model training on compatible GPUs.

### Integration Plan

1. **Model Fine-Tuning**:  
   - A pre-trained GPT-2 model is fine-tuned on a text corpus using PyTorch Lightning and the Hugging Face Transformers library.
   - Training and validation data loaders handle tokenization, padding, and truncation to create high-quality inputs for the model.

2. **Evaluation**:  
   - Evaluation of the model's performance on the validation dataset using established metrics.

3. **Reproducibility**:  
   - Dockerized workflows ensure consistent environments for training and testing.
   - Hugging Face Datasets and Transformers provide reproducible and easy-to-implement data handling.

## Data


The project uses the [**WikiText-2**](https://huggingface.co/datasets/Salesforce/wikitext) dataset, 
which consists of high-quality English text. 
This dataset offers structured, clean, and context-rich content,
making it ideal for text generation tasks.

Preprocessing steps include:
- Removing empty or invalid text samples.
- Tokenizing, truncating, and padding text for input to the model.

## Models

The project is built around the following model architecture:

- **GPT-2 Fine-Tuned Model**:
  - Fine-tuned on the WikiText-2 dataset for English text generation.
  - Lightweight and efficient for deployment on accessible hardware.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yoghurtina/mlops
```
2. Install the required dependencies:
```bash
pipx install invoke
invoke create-environment
invoke requirements
invoke dev-requirements
```
3. Activate the virtual environment:
```bash
conda activate mlops
```
4. Install the pre-commit hooks:
```bash
pre-commit install
```

## Development Guidelines

We use `ruff` to enforce code style and automatically fix issues in the codebase. 
The `invoke lint` command runs ruff with automatic fixing enabled.

- To run linting and automatically fix issues:
```bash
   invoke lint
```
## Training and Evaluation
To fine-tune the GPT-2 model on the WikiText-2 dataset, use the `train` command. 
This command will load the training data, initialize the model, and start the fine-tuning process.

```bash
   train
```
To evaluate the fine-tuned model on the validation set and calculate metrics (e.g., perplexity), use the evaluate command.
```bash
   evaluate
```
- Note: Both commands rely on the project's CLI integration, set up via the `pyproject.toml` file.
Ensure you have installed the project in editable mode before running these commands:
```bash
   pip install -e .
```
### Docker Usage
Use the `train` and `evaluate` commands to fine-tune and evaluate your GPT-2 model, 
either directly through the CLI or via Docker.

#### Build Docker Images

1. Build the training image:  
   `docker build -f dockerfiles/train.dockerfile -t mlops-train .`

2. Build the evaluation image:  
   `docker build -f dockerfiles/evaluate.dockerfile -t mlops-evaluate .`

#### Run Containers

1. **Train the model**:  
   `docker run --rm mlops-train`

2. **Evaluate the model**:  
   `docker run --rm mlops-evaluate`
   
#### Deployment

To deploy on GCP, you can use the following command:
```bash
gcloud run deploy mlops-api --image us-central1-docker.pkg.dev/mlops-448421/mlops-docker-repo/mlops-api --platform managed --region us-central1 --allow-unauthenticated
```

#### Using GPUs

If you have GPU support, run the containers with GPU access:  
`docker run --rm --gpus all mlops-train`


## Project Structure
```
├── .dvc/                     # DVC metadata for versioning data
├── .github/                  # GitHub actions and dependabot configuration
│   └── workflows/
│       ├── ci.yaml           # Continuous integration pipeline
│       ├── container.yml     # Container build pipeline
│       └── gac.yml           # Google Cloud-related actions
├── configs/                  # Configuration files for Hydra
│   ├── __init__.py           # Package initialization
│   └── config.yaml           # Default project configuration
├── dockerfiles/              # Dockerfiles for building images
├── docs/                     # Project documentation
│   ├── mkdocs.yml            # MkDocs configuration for documentation
│   └── source/               # Documentation source files
│       └── index.md
├── jobs/                     # Placeholder for automated job configurations
├── lightning_logs/           # Logs generated during training
├── notebooks/                # Jupyter notebooks for exploration
├── outputs/                  # Outputs from training or inference
├── reports/                  # Reports
├── src/                      # Source code for the project
│   └── mlops/                # Main Python package for the project
│       ├── __init__.py       # Package initialization
│       ├── api.py            # FastAPI app for text generation
│       ├── data.py           # Data loading and preprocessing logic
│       ├── evaluate.py       # Evaluation logic
│       ├── model.py          # Model definitions and utilities
│       ├── quantize.py       # Quantization and pruning logic
│       ├── train.py          # Training script
│       ├── util.py           # Utility functions
│       └── visualize.py      # Visualization utilities
├── tests/                    # Unit tests for the project
├── .coverage                 # Coverage report for tests
├── .coveragerc               # Coverage configuration
├── .dvcignore                # Files to ignore in DVC
├── .gitignore                # Files to ignore in Git
├── .pre-commit-config.yaml   # Pre-commit hooks configuration
├── .ruff_cache/              # Ruff linter cache
├── CHECKLIST.md              # Checklist for project milestones
├── LICENSE                   # Project license
├── models.dvc                # DVC metadata for model versioning
├── pyproject.toml            # Python project configuration
├── README.md                 # Project README
├── requirements.txt          # Runtime dependencies
├── requirements_dev.txt      # Development dependencies
└── tasks.py                  # Invoke tasks for automation

```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

# Notes

The model checkpoint also inlcudes the optimizer state, allowing for resuming training from the last checkpoint.
This also means that the file size will be larger than just the model weights.

gsutil ls -la gs://mlops_team13_bucket/models
gcloud ai custom-jobs create --region=us-central1 --config=jobs/train.yml
gcloud run deploy mlops-api --image us-central1-docker.pkg.dev/mlops-448421/mlops-docker-repo/mlops-api --platform managed --region us-central1 --allow-unauthenticated
