# AI-Powered Customer Support Chatbot

Intelligent financial customer support system with automatic extraction of important user details, conversational memory, and real-time web search.

## Overview

This chatbot uses Google Gemini 2.5 Flash to provide financial support while automatically extracting and remembering user information across sessions. It searches the web for current data and maintains conversation context through an intelligent memory system.

## Details

- **Google Gemini 2.5 Flash:** LLM via LangChain
- **Automatic Entity Extraction:** GLiNER + regex for names, locations, account details, emails, phone numbers
- **Two-Tier Memory System:** Recent history (10 messages) + automatic LLM-based summarization
- **Real-Time Web Search:** DuckDuckGo integration triggered by financial keywords
- **Persistent Storage:** User details saved across sessions in JSON format

## Project Structure
```
task4/
├── chat.py # main conversation handler
├── config.py # configuration settings
├── extract_entities.py # GLiNER entity extraction
├── main.py # CLI interface
├── memory.py # Two tier memory system
├── requirements.txt # python dependencies
├── user_memory.json # user data
├── image,png # converstaion image
└── README.md # this file
```
## Installation

```pip install -r requirements.txt```

## Setup

### Set Your Gemini API Key

Environment Variable:
``export GEMINI_API_KEY="your-api-key-here``

Or create a `.env` file:
```GEMINI_API_KEY=your-api-key-here```

## Usage

### Run the Chatbot

```python main.py```

### Commands

- `exit` - Quit the chatbot
- `clear` - Reset conversation memory and user details


***Example Conversation can be found in the uploaded screenshot***

## System Architecture

### Entity Extraction
Uses GLiNER (`urchade/gliner_small-v2.1`) for zero-shot named entity recognition. Extracts person, location, date, organization, age from text, plus email, phone, and account numbers via regex. Stores 50-character context windows around entities (confidence threshold: 0.5).

### Memory Management
- **Recent:** Last 10 messages kept in full
- **Compression:** Older messages auto-summarized by LLM (100 words max)
- **Persistence:** User details saved to `user_memory.json` for multi-session recall

### Web Search
Triggers DuckDuckGo search when queries contain financial keywords (`rate`, `price`, `stock`, `market`, `news`, `current`, `latest`, `today`, `company`). Top 400 characters injected into context.

### Conversation Flow
1. Extract entities → store
2. Check keywords → search if needed
3. Build context (user details + history + search results)
4. Query Gemini 2.5 Flash → respond

## Configuration

Edit `config.py`:
```
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.7
RECENT_MESSAGE_LIMIT = 10
SEARCH_KEYWORDS = ["rate", "price", "stock", "market", "news", "current", "latest", "today", "company"]
