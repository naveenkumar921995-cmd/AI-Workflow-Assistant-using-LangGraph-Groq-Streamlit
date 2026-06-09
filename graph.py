from typing import TypedDict
import streamlit as st

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq


class State(TypedDict):
    messages: list
    sentiment: str
    research: str
    answer: str
    summary: str


# -------------------
# Load Groq Key
# -------------------

if "GROQ_API_KEY" not in st.secrets:
    raise ValueError("GROQ_API_KEY not found in Streamlit Secrets")

llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.3-70b-versatile"
)


# -------------------
# Preprocess Node
# -------------------

def preprocess(state: State):
    state["messages"][-1].content = (
        state["messages"][-1].content.strip()
    )
    return state


# -------------------
# Sentiment Node
# -------------------

def analyze_sentiment(state: State):

    msg = state["messages"][-1].content.lower()

    positive_words = [
        "good",
        "great",
        "excellent",
        "awesome",
        "happy",
        "amazing",
    ]

    negative_words = [
        "bad",
        "sad",
        "terrible",
        "angry",
        "worst",
    ]

    if any(word in msg for word in positive_words):
        state["sentiment"] = "Positive"

    elif any(word in msg for word in negative_words):
        state["sentiment"] = "Negative"

    else:
        state["sentiment"] = "Neutral"

    return state


# -------------------
# Research Node
# -------------------

def search_node(state: State):

    query = state["messages"][-1].content

    prompt = f"""
Provide important research information about:

{query}

Give concise factual information.
"""

    research = llm.invoke(prompt)

    state["research"] = research.content

    return state


# -------------------
# Answer Node
# -------------------

def answer_node(state: State):

    prompt = f"""
User Query:
{state["messages"][-1].content}

Research:
{state["research"]}

Generate a detailed answer.
"""

    answer = llm.invoke(prompt)

    state["answer"] = answer.content

    return state


# -------------------
# Summary Node
# -------------------

def summary_node(state: State):

    prompt = f"""
Summarize the following in 3 bullet points:

{state["answer"]}
"""

    summary = llm.invoke(prompt)

    state["summary"] = summary.content

    return state


# -------------------
# Chatbot Node
# -------------------

def chatbot(state: State):

    state["messages"].append(
        AIMessage(content=state["answer"])
    )

    return state


# -------------------
# Logger Node
# -------------------

def logger(state: State):

    print("\n===== WORKFLOW LOG =====")
    print("Sentiment:", state["sentiment"])
    print("Research Generated")
    print("Answer Generated")
    print("Summary Generated")

    return state


# -------------------
# Build Graph
# -------------------

builder = StateGraph(State)

builder.add_node("preprocess", preprocess)
builder.add_node("sentiment", analyze_sentiment)
builder.add_node("research", search_node)
builder.add_node("answer", answer_node)
builder.add_node("summary", summary_node)
builder.add_node("chatbot", chatbot)
builder.add_node("logger", logger)

builder.add_edge(START, "preprocess")
builder.add_edge("preprocess", "sentiment")
builder.add_edge("sentiment", "research")
builder.add_edge("research", "answer")
builder.add_edge("answer", "summary")
builder.add_edge("summary", "chatbot")
builder.add_edge("chatbot", "logger")
builder.add_edge("logger", END)

graph = builder.compile()
