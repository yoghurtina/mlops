# MLOps: Large Language Model (LLM) for the Greek Language

## Project Description

### Overall Goal of the Project
The goal of this project is to develop a small Large Language Model (LLM) tailored for the Greek language. 
The model will support tasks such as text generation and natural language understanding in Greek. 
By focusing on Greek, this project aims to fill the gap in AI tools for smaller language communities. 

We aim to use MLOps practices to make development efficient and reproducible.
The model will be designed to be lightweight, comparable to GPT-2, 
making it feasible to train on accessible hardware.

### Framework and Integration
We will use the following frameworks and tools:
- **PyTorch**: Selected for its flexibility in custom model development and widespread use in the deep learning community, making it easier to find resources and support.
- **Hugging Face Transformers**: Provides a robust library of pre-trained models, which will save time and effort in fine-tuning models specifically for Greek tasks.
- **Docker**: Ensures a consistent and reproducible setup by containerizing the environment, simplifying collaboration and deployment.
- **Git**: Used for version control, enabling collaboration and maintaining a clear history of changes.
- **Code Environments**: Tools such as Visual Studio Code (VSCode) and Vim for writing and debugging code.
- **Deep Learning Software**: Such as CUDA and cuDNN to accelerate model training on GPUs.

Integration Plan:
1. Fine-tune a pre-trained model using Hugging Face Transformers.
2. Use Docker to ensure a consistent and reproducible setup.
3. Automate workflows for training and deployment.

### Data
We will use the following datasets:

- **[Greek Wikipedia](https://huggingface.co/datasets/legacy-datasets/wikipedia?utm_source=chatgpt.com)**: Wikipedia provides language-specific dumps, 
including Greek. These dumps contain high-quality, structured articles in Greek, suitable for various natural language processing tasks.

- **[OpenSubtitles](https://paperswithcode.com/dataset/opensubtitles?utm_source=chatgpt.com)**: OpenSubtitles is a collection of multilingual parallel corpora compiled from a large database of movie and TV subtitles. The dataset includes subtitles in Greek, offering conversational and informal text that can be valuable for training language models.

These datasets will be preprocessed to clean and tokenize the text, ensuring high-quality inputs.

### Models
We will focus on the following models:
- **GPT-2-based Model**: A small, manageable model fine-tuned for Greek text generation and completion.
- **BERT-based Model**: Adapted for Greek tasks like sentiment analysis and question answering.
An LLM for the Greek language

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
└── tasks.py                  # Project tasks
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
