# Chatbot with Transformers

# GPT-2 Based Chatbot for Git and GitHub Assistance

A GPT-2 based chatbot implementation focused on Git and GitHub assistance, built using Hugging Face's Transformers library.

## Overview

This project implements a conversational AI assistant using transformer architecture, specifically fine-tuned on Git/GitHub related queries. The model is built using PyTorch and Hugging Face's Transformers library.

## Features

- Fine-tuned GPT-2 model
- Memory-optimized training
- Mixed precision support
- Comprehensive error handling
- Conversation history management

## Dataset

- 300+ curated Q&A pairs
- Covers Git commands, GitHub operations, and common workflows
- Multiple complexity levels (beginner to advanced)
- [View dataset](data/chatbot.csv)

## Technical Stack

- PyTorch
- Hugging Face Transformers
- Streamlit
- CUDA support for GPU acceleration

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/thedavidemmanuel/chatbot-with-transformers.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the Streamlit interface
streamlit run src/app.py
```

## Development Process

See `notebooks/chatbot_development.ipynb` for the complete development process including:

- Data preprocessing
- Model architecture
- Training process
- Performance metrics

## Model Performance

- **Training Loss:** X
- **Validation Loss:** Y
- **Response Accuracy:** Z%
- **BLEU Score:** W

## Demo

[Link to demo video](https://www.loom.com/share/979d2fd225c9424e97530482f675ca4f?sid=6ad974f6-cc34-4c6c-b349-eb2e78e50d33)

## License

This project is licensed under the MIT License.

## Author

David Emmanuel
