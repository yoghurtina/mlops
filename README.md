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

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Invoke tasks
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
