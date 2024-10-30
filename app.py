import streamlit as st
import numpy as np
import json
import pickle
import random
import torch
import transformers
from transformers import BertModel, BertTokenizer
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

# Page configuration
st.set_page_config(
    page_title="GitHub Helper Bot",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for better chat appearance
st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stChatInput {
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('wordnet')

# Initialize BERT
@st.cache_resource
def load_bert():
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        return tokenizer, bert_model
    except Exception as e:
        st.error(f"Error loading BERT models: {str(e)}")
        return None, None

# Load the saved files
@st.cache_resource
def load_files():
    try:
        model = load_model('github_chatbot_model.h5')
        with open('github_words.pkl', 'rb') as f:
            words = pickle.load(f)
        with open('github_classes.pkl', 'rb') as f:
            classes = pickle.load(f)
        with open('github_intents.json', 'r') as f:
            intents = json.load(f)
        return model, words, classes, intents
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None, None, None

def get_bert_embedding(sentence, tokenizer, bert_model):
    try:
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    except Exception as e:
        st.error(f"Error getting BERT embeddings: {str(e)}")
        return None

def predict_class(sentence, model, classes, tokenizer, bert_model):
    embedding = get_bert_embedding(sentence, tokenizer, bert_model)
    if embedding is None:
        return []
    
    try:
        res = model.predict(embedding)[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
        return return_list
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return []

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure how to help with that. Could you rephrase your question?"
    try:
        tag = intents_list[0]['intent']
        for i in intents_json['intents']:
            if tag in i['tags']:
                return random.choice(i['responses'])
        return "I don't understand. Could you try asking in a different way?"
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")
        return "I encountered an error. Please try again."

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your GitHub Helper Bot. How can I assist you today?"}
    ]

# Download NLTK data
download_nltk_data()

# Header
st.title("GitHub Helper Bot 🤖")

# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This bot can help you with:
    - Creating repositories
    - Managing pull requests
    - Git commands
    - Issues and forks
    - And more!
    """)
    
    if st.button("Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your GitHub Helper Bot. How can I assist you today?"}
        ]
        st.rerun()

# Load models
try:
    tokenizer, bert_model = load_bert()
    model, words, classes, intents = load_files()

    if None in (tokenizer, bert_model, model, words, classes, intents):
        st.error("Failed to load required models and files.")
        st.stop()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about GitHub..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ints = predict_class(prompt, model, classes, tokenizer, bert_model)
                response = get_response(ints, intents)
                
                # Show confidence if available
                if ints:
                    confidence = float(ints[0]['probability'])
                    if confidence > 0.9:
                        st.markdown(response)
                    else:
                        st.markdown(f"{response}\n\n*Confidence: {confidence:.1%}*")
                else:
                    st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Made with ❤️ using Streamlit and BERT*")