# Sample PS : Task Submission
## Overview
This repository contains implementations of my submission for the Pathway sample PS.
The problem statement included four tasks, out of which Task 1 is compulsory, and at least one of Tasks 2–4 must be completed.
I have completed Task 1 (Docker setup), Task 2 (Real-Time AI-Driven Stock Price Prediction) and Task 4 (AI-Powered Customer Support).

## Repository Structure

```
Pathway-sample-ps/
├── task1/   # Pathway Docker deployment
├── task2/   # Bitcoin LSTM prediction model
├── task4/   # AI customer support system
└── README.md
```


## Task 1: Docker Setup & Pathway Deployment

Containerized the Pathway project using a simple Python-based Dockerfile and verified successful execution inside the container.

## Task 2: Bitcoin Price Prediction

### Details

- **LSTM:** Two-layer architecture (64→32 units) with dropout regularization
- **Technical Indicators:** MA5, MA10, MA20, RSI14, volatility, price range, returns
- **Data:** Used historical Bitcoin prices (Nov 2013 – Dec 2024 for training, Jan 2025 – Oct 2025 for testing)
- **Prediction:** Outputs the probability of upward movement; converted to next-step price direction
- **Evaluation:** Directional accuracy ~54.7%, showing the model can follow short-term trends without looking at future data
 
## Task 4: AI-Powered Customer Support

### Details

- **LLM** Google Gemini 2.5 Flash via LangChain
- **Automatic Entity Extraction:** GLiNER + regex for names, locations, account numbers, emails, etc.
- **Two-Tier Memory:** Recent history (10 messages) + automatic summarization
- **Real-Time Web Search:** DuckDuckGo integration for current financial data
- **Persistent Storage:** JSON-based user details across sessions


Each task is contained in its respective directory with:
- Installation instructions (`requirements.txt`)
- Detailed README

See individual task folders for detailed setup and usage instructions.
