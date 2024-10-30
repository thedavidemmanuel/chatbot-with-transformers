# GitHub Support Chatbot Using BERT

## BSE - ML Techniques I Assignment

### Overview

A BERT-based chatbot implementation trained to handle GitHub-related queries. The chatbot leverages transformer architecture to understand and respond to user questions about Git and GitHub operations.

### Dataset & Preprocessing

- **Source**: Custom GitHub support dataset (`data/chatbot.csv`)
- **Size**: 500+ conversation pairs
- **Processing**:
  - BERT WordPiece tokenization
  - NLTK lemmatization
  - Noise removal and normalization
  - Proper formatting for transformer input

### Model Architecture & Training

#### Components

- **Base Model**: BERT (bert-base-uncased)
- **Additional Layers**:
  - Dense Layer (128 units, ReLU)
  - Dropout (0.2)
  - Dense Layer (64 units, ReLU)
  - Dropout (0.2)
  - Output Layer (Softmax)

#### Hyperparameters

- Learning Rate: 0.0001
- Batch Size: 12
- Epochs: 50
- Optimizer: Adam

### Performance Metrics

- **Training**:

  - Accuracy: 98.77%
  - Loss: 0.0446
  - Precision: 0.9918
  - Recall: 0.9877

- **Validation**:
  - Accuracy: 72.46%
  - Loss: 1.1667
  - F1 Score: 0.7248
  - Precision: 0.7692
  - Recall: 0.7246

### Project Structure

```
chatbot-with-transformers/
├── data/
│   └── gitbotdataset.csv         # Training dataset
├── app.py                  # Streamlit interface
├── gitbot.ipynb           # Training notebook
├── github_chatbot_model.h5
├── github_classes.pkl
├── github_intents.json
├── github_words.pkl
├── README.md
└── requirements.txt
```

### Live Demo & Examples

- **Live Demo**: [GitHub Helper Bot on Streamlit](https://chatbot-with-transformers.streamlit.app/)
- **Demo Video**: [Link to demo video](https://www.loom.com/share/979d2fd225c9424e97530482f675ca4f?sid=6ad974f6-cc34-4c6c-b349-eb2e78e50d33)

#### Example Interactions

```
User: How do I create a repository?
Bot: To create a new repository on GitHub:
1. Click the '+' icon in the top right
2. Select 'New repository'
3. Fill in repository name and details
4. Click 'Create repository'

User: What is a pull request?
Bot: A pull request (PR) is a way to propose changes to a repository. It lets you:
- Show others your code changes
- Get feedback through code review
- Discuss modifications
- Merge changes when approved
```

### Setup Instructions

1. Clone repository:

```bash
git clone https://github.com/thedavidemmanuel/chatbot-with-transformers.git
cd chatbot-with-transformers
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run locally:

```bash
streamlit run app.py
```

### Contributors

- David Emmanuel

### License

MIT License

---

_Note: This project was developed as part of the ML Techniques I course at African Leadership University._

```

```
