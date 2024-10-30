# GitHub Support Chatbot Using BERT

## BSE - ML Techniques I Assignment

### Overview

A BERT-based chatbot implementation specifically trained to handle GitHub-related queries and provide technical assistance. The chatbot leverages transformer architecture and deep learning to understand and respond to user questions about Git and GitHub operations.

### Dataset

- **Source**: Custom GitHub support dataset (`data/chatbot.csv`)
- **Coverage**:
  - Git commands and operations
  - GitHub platform features
  - Repository management
  - Collaboration workflows
  - Common troubleshooting scenarios

### Technical Architecture

#### Model Components

- Base Model: BERT (bert-base-uncased)
- Additional Layers:
  - Dense Layer (128 units, ReLU)
  - Dropout (0.2)
  - Dense Layer (64 units, ReLU)
  - Dropout (0.2)
  - Output Layer (Softmax)

### Project Structure

```
chatbot-with-transformers/
├── data/
│   └── chatbot.csv
├── app.py                    # Streamlit interface
├── gitbot.ipynb             # Training notebook
├── github_chatbot_model.h5  # Trained model
├── github_classes.pkl       # Saved classes
├── github_intents.json      # Intent configurations
├── github_words.pkl         # Processed words
├── README.md
└── requirements.txt         # Project dependencies
```

### Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/thedavidemmanuel/chatbot-with-transformers.git
cd chatbot-with-transformers
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Run the Streamlit app locally:

```bash
streamlit run app.py
```

Example interactions:

```
You: How do I create a repository?
Bot: To create a new repository on GitHub:
1. Click the '+' icon in the top right
2. Select 'New repository'
3. Fill in repository name and details
4. Click 'Create repository'

You: What is a pull request?
Bot: A pull request (PR) is a way to propose changes to a repository. It lets you:
- Show others your code changes
- Get feedback through code review
- Discuss modifications
- Merge changes when approved
```

### Live Demo

Try the chatbot: [GitHub Helper Bot on Streamlit](https://chatbot-with-transformers.streamlit.app/)

[Link to demo video](https://www.loom.com/share/979d2fd225c9424e97530482f675ca4f?sid=6ad974f6-cc34-4c6c-b349-eb2e78e50d33)

### Model Performance

- Training Accuracy: 98.77%
- Validation Accuracy: 72.46%
- F1 Score: 0.7248

### Contributors

- David Emmanuel

### License

This project is licensed under the MIT License.
