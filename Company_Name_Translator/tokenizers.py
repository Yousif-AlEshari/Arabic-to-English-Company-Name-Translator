from typing import List

try:
    import spacy
    _spacy_en = spacy.load("en_core_web_sm")
    def tokenize_en(text: str) -> List[str]:
        return [t.text for t in _spacy_en.tokenizer(text)]
except Exception:
    _spacy_en = None
    def tokenize_en(text: str) -> List[str]:
        return text.strip().split()

def tokenize_ar(text: str) -> List[str]:
    return text.strip().split()
