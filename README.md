# Large Language Model (LLM) for the Greek Language

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
- **PyTorch**: For training and fine-tuning the model.
- **Hugging Face Transformers**: For leveraging pre-trained models as a base for fine-tuning.
- **Docker**: To containerize the environment for easy deployment.
- **Git**: For version control and collaboration.
- **Code Environments**: Tools like Visual Studio Code (VSCode) and Vim for writing and debugging code.
- **Deep Learning Software**: Such as CUDA and cuDNN to accelerate model training on GPUs.

Integration Plan:
1. Fine-tune a pre-trained model using Hugging Face Transformers.
2. Use Docker to ensure a consistent and reproducible setup.
3. Automate workflows for training and deployment.

### Data
We will use the following datasets:
- **Greek Common Crawl Dataset**: A large corpus of web text in Greek.
- **Greek Wikipedia**: For high-quality, structured text.
- **OpenSubtitles**: To add conversational and informal text.

These datasets will be preprocessed to clean and tokenize the text, ensuring high-quality inputs.

### Models
We will focus on the following models:
- **GPT-2-based Model**: A small, manageable model fine-tuned for Greek text generation and completion.
- **BERT-based Model**: Adapted for Greek tasks like sentiment analysis and question answering.
