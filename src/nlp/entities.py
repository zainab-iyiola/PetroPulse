from __future__ import annotations

import json
from typing import Dict, List, Tuple
import spacy

# Load spaCy model once (shared across calls)
# en_core_web_sm is light and installs quickly
try:
    _NLP = spacy.load("en_core_web_sm")
except OSError:
    # Safety: allow the app to run even if model missing
    _NLP = None

ORG_LABELS = {"ORG"}            # companies, orgs
GPE_LABELS = {"GPE"}            # countries, cities

def extract_org_gpe(text: str, title: str = "") -> Tuple[List[str], List[str]]:
    """Return unique ORG and GPE mentions from text (and title as a booster)."""
    if not text and not title or _NLP is None:
        return [], []

    doc = _NLP(f"{title}\n{text}") if text else _NLP(title)

    orgs = []
    gpes = []
    for ent in doc.ents:
        if ent.label_ in ORG_LABELS:
            orgs.append(ent.text.strip())
        elif ent.label_ in GPE_LABELS:
            gpes.append(ent.text.strip())

    # normalize/dedupe while preserving order
    def _unique(seq):
        seen = set()
        out = []
        for s in seq:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    return _unique(orgs), _unique(gpes)

def entities_to_json(orgs: List[str], gpes: List[str]) -> str:
    """Store as JSON string in DB column `entities`."""
    return json.dumps({"org": orgs, "gpe": gpes})
