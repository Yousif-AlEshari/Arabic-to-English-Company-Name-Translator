import re
from org_renderer import render_org_name_en
from model import greedy_translate
from translit import ar_normalize

ORG_KEYS = {"شركة","شركه","مجموعة","بنك","مصرف","قابضة","القابضة","قابضه","ش.م.ع","ش م ع","ش.ذ.م.م","ش ذ م م"}
def is_arabic_char(c): return '\u0600' <= c <= '\u06FF' or c in {' ', '.'}

def is_likely_name(text: str) -> bool:
    s = ar_normalize(text)
    if not s: return False
    ratio = sum(1 for ch in s if is_arabic_char(ch)) / max(1, len(s))
    if ratio < 0.6: return False
    if any(k in s for k in ORG_KEYS): return True
    toks = s.split()
    return len(toks) <= 8 and not re.search(r"[؟\?\!\;\,\:]", s)

def translate_smart(model, sentence_ar: str, src_vocab, trg_vocab, device, max_len=50):
    if is_likely_name(sentence_ar):
        return render_org_name_en(sentence_ar)
    return greedy_translate(model, sentence_ar, src_vocab, trg_vocab, device, max_len=max_len)
