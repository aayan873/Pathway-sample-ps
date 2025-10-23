import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config import MEMORY_FILE, RECENT_MESSAGE_LIMIT
import os

class SmartConversationMemory:
    def __init__(self, llm, recent_limit=RECENT_MESSAGE_LIMIT):
        self.llm = llm
        self.recent_history = []
        self.summary = ""
        self.recent_limit = recent_limit
        
    def add(self, user_msg, ai_msg):
        # add the message exchanged between bot and user

        self.recent_history.append(HumanMessage(content=user_msg))
        self.recent_history.append(AIMessage(content=ai_msg))
        
        # if length of history is more than limit we compress
        if len(self.recent_history) > self.recent_limit:
            self.compress()
    
    def compress(self):
        # For compression we summarize old messages

        to_sum = self.recent_history[:4]
        self.recent_history = self.recent_history[4:]
        
        summary_txt = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
            for msg in to_sum
        ])
        
        prompt = f"""Summarize this conversation briefly, keeping all the key details:
        {summary_txt}

        Previous summary: {self.summary if self.summary else 'None'}

        Concise summary (max 100 words):"""
    
        sum = self.llm.invoke([HumanMessage(content=prompt)])
        self.summary = sum.content.strip()
    
    def get_context(self):
        # returns sumarry as well as prev messages

        context = []
        
        if self.summary:
            context.append(SystemMessage(content=f"Previous conversation summary: {self.summary}"))
        
        context.extend(self.recent_history)
        return context
    
    def clear(self):
        # Clears memory
        self.recent_history = []
        self.summary = ""


class UserDetails:
    # Stores user info

    def __init__(self, file_path=MEMORY_FILE):
        self.file_path = file_path
        self.data = self.load()
    
    def load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def store(self, key, value, context=None):
        # Store entity and context

        if context:
            self.data[key] = {
                "value": value,
                "context": context
            }
        else:
            self.data[key] = value
            
        self.save()
    
    def get_context(self):
        if not self.data:
            return "No user details stored yet."
        
        details = []
        for key, data in self.data.items():
            if isinstance(data, dict) and "value" in data:
                # Has context
                details.append(f"{key}: {data['value']} (context: {data['context']})")
            else:
                # Simple value
                details.append(f"{key}: {data}")
        
        return "User details:\n" + "\n".join(details)
    
    def clear(self):
        self.data = {}
        self.save()