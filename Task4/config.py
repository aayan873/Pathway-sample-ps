import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("GEMINI_API_KEY not set")
    exit(1)

# Memory Settings
MEMORY_FILE = "user_memory.json"
RECENT_MESSAGE_LIMIT = 10  # Keep last 5 exchanges raw
SUMMARY_TOKEN_LIMIT = 500  # Max tokens for summarizing history

# Search Keywords
SEARCH_KEYWORDS = ["rate", "price", "stock", "market", "news", "current", "latest", "today", "company"]

# Model Settings
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.7
MAX_RETRIES = 3
