import re
from gliner import GLiNER

# Load GLiNER model
print("Loading GLiNER model...")
ner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
print("GLiNER model loaded!\n")

def get_context_window(text, entity_text, window_size=50):
    # returns content around entity

    start_pos = text.lower().find(entity_text.lower())
    if start_pos == -1:
        return entity_text
    
    # gets before + entity + after
    context_start = max(0, start_pos - window_size)
    context_end = min(len(text), start_pos + len(entity_text) + window_size)
    
    context = text[context_start:context_end].strip()
    return context


def extract(text):

    result = {}
    
    labels = ["person", "location", "city", "date", "organization", "event", "age"]
    
    entities = ner_model.predict_entities(text, labels, threshold=0.5)
    
    for entity in entities:
        entity_text = entity["text"].strip()
        entity_label = entity["label"].lower()
        
        # skip if already present
        if entity_label in result:
            continue
        
        # filter
        if len(entity_text) >= 3:
            result[entity_label] = entity_text
            result[f"{entity_label}_context"] = get_context_window(text, entity_text)

    # Regex for ensuring other user info
    email_match = re.search(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', text)
    if email_match:
        result["email"] = email_match.group(0)
    
    account_match = re.search(r'account\s+(?:number\s+)?(?:is\s+)?(\d{6,16})', text, re.IGNORECASE)
    if account_match:
        result["account_number"] = account_match.group(1)
    
    if "account_number" not in result:
        phone_match = re.search(r'(?:phone|mobile|cell|contact).*?(\+?\d{7,15})', text, re.IGNORECASE)
        if not phone_match:
            phone_match = re.search(r'\b(\+?\d{1,3}[\s\-]?)\d{3}[\s.\-]?\d{3}[\s.\-]?\d{4}\b', text)
        if phone_match:
            result["phone"] = phone_match.group(1) if phone_match.lastindex else phone_match.group(0)
    
    return result
