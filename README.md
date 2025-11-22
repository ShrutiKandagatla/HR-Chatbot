# HR-Chatbot

A lightweight retrieval-based chatbot for HR & payroll queries. Built as a proof-of-concept to combine domain knowledge (HR/payroll) with NLP retrieval techniques.


## Features
- FAQ-based retrieval using TF-IDF + cosine similarity
- Small rule-based handlers for common tasks (leave balance, payslip link simulation)
- Streamlit UI for quick demo
- Easy to extend with embeddings (sentence-transformers) or generative models


## Tech Stack
- Python 3.8+
- scikit-learn (TfidfVectorizer)
- pandas
- Streamlit


## Setup
1. Clone the repo
2. Create a virtual environment


```bash
python -m venv venv
source venv/bin/activate # on mac/linux
venv\Scripts\activate # on Windows
pip install -r requirements.txt
