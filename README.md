# AI Workflow Assistant using LangGraph & Groq

## Overview

This project demonstrates an AI workflow pipeline built using LangGraph and Groq's Gemma2-9B model.

The application processes user queries through multiple intelligent stages:

- Preprocessing
- Sentiment Analysis
- Research Generation
- Answer Generation
- Summarization
- Chatbot Response
- Logging

## Tech Stack

### Frontend

- Streamlit

### Backend

- Python

### AI Framework

- LangGraph
- LangChain

### LLM

- Groq
- Gemma2-9B-It

## Features

- Modular graph-based workflow
- State management using TypedDict
- AI-powered answer generation
- Research and summarization
- Interactive Streamlit dashboard
- Cloud deployment support

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Deployment

Deploy directly on Streamlit Cloud.

Add Secret:

```toml
GROQ_API_KEY="your_key"
```

## Skills Demonstrated

- Generative AI
- LangGraph
- LangChain
- Streamlit
- NLP
- Workflow Automation
- Prompt Engineering
- State Management
