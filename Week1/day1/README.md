# Day 1 – Introduction to LLM APIs

## Objective

Learn how to connect a Python application to a Large Language Model using the Groq API.

---

## Topics Covered

- Python Virtual Environment (`.venv`)
- Environment Variables using `.env`
- API Keys
- Installing packages with `uv`
- Groq Python SDK
- Sending prompts to an LLM
- Understanding:
  - Client
  - Model
  - Messages
  - Role
  - Response

---

## Project

A simple Python program that:

- Loads the API key securely
- Creates a Groq client
- Sends a prompt
- Prints the generated response

Example prompt:

> "Write a short poem about the beauty of nature."

---

## Concepts Learned

### Virtual Environment

Keeps project dependencies isolated from other Python projects.

---

### Environment Variables

API keys should never be hardcoded.


### Commands Used:
1. uv init day1
2. uv venv --python 3.11
3. .venv\Scripts\activate
4. deactivate
5. code hello_llm.py
6. uv add groq python-dotenv
7. python .\hello_llm.py

---