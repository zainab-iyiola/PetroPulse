import re
from typing import List
import spacy
from nltk.corpus import stopwords

# Try loading spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# Load stopwords for cleaning
try:
    STOP = set(stopwords.words("english"))
except Exception:
    STOP = set()

def clean_text(text: str) -> str:
    """
    Remove extra spaces and normalize text.
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    return text

def extract_entities(text: str) -> List[str]:
    """
    Extract named entities using spaCy (ORG, GPE, etc.).
    """
    if not nlp:
        return []
    doc = nlp(text)
    ents = [f"{ent.label_}:{ent.text}" for ent in doc.ents]
    return ents[:50]  # Limit to 50 entities
