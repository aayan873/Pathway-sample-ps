import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
from config import GEMINI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_RETRIES, SEARCH_KEYWORDS
from memory import SmartConversationMemory, UserDetails
from extract_entities import extract

# initialize LLM
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=TEMPERATURE,
    max_retries=MAX_RETRIES
)

# initialize ddgs
search_tool = DuckDuckGoSearchRun()

# initialize memory
memory = SmartConversationMemory(llm)
user_details = UserDetails()


def should_search(text):
    #checks if websearch is needed

    return any(kw in text.lower() for kw in SEARCH_KEYWORDS)


def chat(user_text):
    # Main chat function

    # extract and store entities
    extracted = extract(user_text)
    stored_count = 0
    for key, value in extracted.items():
        if "_context" not in key:
            # check if there's a context for this entity
            context_key = f"{key}_context"
            context = extracted.get(context_key, None)
            
            # store with context
            user_details.store(key, value, context)
            stored_count += 1

    if stored_count > 0:
        entity_names = [k for k in extracted.keys() if "_context" not in k]
        print(f"  [Stored: {', '.join(entity_names)}]")

    # web search if needed
    search_results = ""
    if should_search(user_text):
        print("  [Searching web...]")
        try:
            search_results = search_tool.run(user_text)
            search_results = f"\n\nWeb search results:\n{search_results[:400]}"
        except:
            search_results = "\n\n(Web search unavailable)"
    
    # get context
    user_context = user_details.get_context()
    
    # build messages
    messages = [
        SystemMessage(content=f"""You are an autonomous AI financial support assistant.

    {user_context}
    {search_results}

    Guidelines:
    - Be professional, friendly, and concise
    - Remember and use information from previous messages
    - Answer questions naturally
    - If a task requires human intervention, say "You'll need to contact your bank directly"

    Respond naturally - like a knowledgeable assistant, not a call center script.""")
    ]
    
    # add memory context
    messages.extend(memory.get_context())
    
    # add current message
    messages.append(HumanMessage(content=user_text))
    
    # get response
    for attempt in range(MAX_RETRIES):
        try:
            response = llm.invoke(messages)
            answer = response.content
            
            # save to memory
            memory.add(user_text, answer)
            
            return answer
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                print(f"[Retrying in {wait_time}s...]")
                time.sleep(wait_time)
            else:
                return "Please try again."


def clear_memory():
    """Clear all memory"""
    user_details.clear()
    memory.clear()
