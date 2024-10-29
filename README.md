# GitHub Support Chatbot Using BERT
## BSE - ML Techniques I Assignment

### Overview
A BERT-based chatbot implementation specifically trained to handle GitHub-related queries and provide technical assistance. The chatbot leverages transformer architecture and deep learning to understand and respond to user questions about Git and GitHub operations.

### Dataset
- **Source**: Custom GitHub support dataset (`gitbotdataset.csv`)
- **Size**: 500+ conversation pairs
- **Distribution**: Balanced across different intents
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

#### Dependencies
```bash
numpy==1.21.0
pandas==1.3.0
torch==1.9.0
transformers==4.11.0
tensorflow==2.6.0
scikit-learn==0.24.2
nltk==3.6.3
matplotlib==3.4.3
seaborn==0.11.2
```

### Data Preprocessing
1. **Text Cleaning**
   - Tokenization using BERT WordPiece tokenizer
   - Lemmatization using NLTK
   - Removal of special characters and noise
   - Handling of missing values

2. **Feature Extraction**
   - BERT embeddings generation
   - Intent classification preparation
   - Label binarization

### Model Performance
#### Training Metrics
- Final Training Accuracy: 98.77%
- Final Training Loss: 0.0446
- Final Precision: 0.9918
- Final Recall: 0.9877

#### Validation Metrics
- Validation Accuracy: 72.46%
- Validation Loss: 1.1667
- F1 Score: 0.7248
- Precision: 0.7692
- Recall: 0.7246

### Installation and Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/github-support-chatbot.git
cd github-support-chatbot
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
1. Start the chatbot interface:
```bash
python chatbot.py
```

2. Example interactions:
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

### Development Process
1. **Data Collection**
   - Curated GitHub-specific conversation pairs
   - Ensured coverage of common user queries
   - Validated response accuracy

2. **Model Training**
   - Implemented progressive learning rate decay
   - Applied dropout for regularization
   - Monitored validation metrics to prevent overfitting

3. **Optimization**
   - Hyperparameter tuning:
     - Learning rate: 0.0001
     - Batch size: 12
     - Epochs: 50
   - Achieved 10%+ improvement over baseline

### Code Structure
```
github-support-chatbot/
├── data/
│   ├── gitbotdataset.csv
│   └── github_intents.json
├── models/
│   └── github_chatbot_model.h5
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── chatbot.py
│   └── utils.py
├── notebooks/
│   └── development.ipynb
├── requirements.txt
└── README.md
```

### Demo
[Link to Demo Video]([https://example.com/demo](https://www.loom.com/share/979d2fd225c9424e97530482f675ca4f?sid=6ad974f6-cc34-4c6c-b349-eb2e78e50d33))


### Contributors
- David Emmanuel

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Note: This project was developed as part of the ML Techniques I course at African Leadership University.*
