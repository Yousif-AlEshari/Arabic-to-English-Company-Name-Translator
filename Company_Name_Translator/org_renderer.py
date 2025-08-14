import re
from translit import ar_normalize, AR_DIACRITICS, BASE
# Helpers
AR_LETTERS = re.compile(r"[\u0600-\u06FF]+")
def strip_diacritics(s: str) -> str: return re.sub(f"[{AR_DIACRITICS}]", "", s)
def has_al_prefix(tok: str) -> bool: return tok.startswith("ال")
def has_ll_prefix(tok: str) -> bool: return tok.startswith("لل") or tok.startswith("لﻟ")
def drop_al(tok: str) -> str: return tok[2:] if has_al_prefix(tok) else tok
def drop_ll(tok: str) -> str: return tok[2:] if tok.startswith("لل") else tok
def cap_words(ws): return " ".join(w[:1].upper() + w[1:] if w else w for w in ws)

# Lexical and business maps
LEXICAL_MAP = {
    "القصر":"Qasr","القصواء":"Qaswa","الملك":"Al-Malik","الخليج":"Al-Khalij","النيل":"Al-Nil",
    "الشاطئ":"Al-Shati'","البعيد":"Al-Ba'eed","الشبكات":"Al-Shabakat","الذكية":"Al-Dhakiyya",
    "الذهبي":"Al-Dhahabi","زهرة":"Zahrat","جوهرة":"Jawharat","عهد":"Ahd","اليقين":"Al-Yaqin",
    "فيض":"Fayd","السعودية":"Al-Saudiya","ارض":"Ard","البروج":"Al-Barouj",
    # "يونيكوم":"Unicom","سمفوني":"Samfoni","سيكشنز":"Sikhanz",
}
BUSINESS_MAP = {
    "شركة":"Company","شركه":"Company","مجموعة":"Group","مكتب":"Office","بنك":"Bank","مصرف":"Bank","قابضة":"Holding","القابضة":"Holding",
    "تجارة":"Trading","التجارة":"Trading","للتجارة":"Trading","لتجارة":"Trading","لتجاره":"Trading","للتجاره":"Trading",
    "العامة":"Public","العامه":"Public",
    "المقاولات":"Contracting","للمقاولات":"Contracting",
    "الاتصالات":"Telecommunications","للاتصالات":"Telecommunications",
    "الخدمات":"Services","لخدمات":"Services","لخدمات الدفع":"Payment Services","الدفع":"Payment",
    "البناء":"Construction",
}
ORG_TYPE_TOKENS = {"شركة","شركه","مجموعة","مكتب","بنك","مصرف","قابضة","القابضة"}

def translit_simple(tok: str) -> str:
    al = ""
    if has_al_prefix(tok):
        tok_wo = drop_al(tok); al = "Al-"
    else:
        tok_wo = tok
    out = "".join(BASE.get(ch, ch) for ch in tok_wo)
    out = re.sub(r"\b(q|k|m|b|f|s|d|t|r|n|l)([bcdfghjklmnpqrstvwxyz])", r"\1a\2", out)
    out = re.sub(r"\'(?=[aeiou])", "", out).replace("--","-").strip("-")
    out = out.capitalize()
    return (al + out) if al else out

def normalize_tokens_ar(text: str) -> list:
    s = ar_normalize(text)
    s = strip_diacritics(s)
    return s.split()

def render_org_name_en(text: str) -> str:
    toks = normalize_tokens_ar(text)
    body_en, business_bits, org_suffix = [], [], []
    for tok in toks:
        t = tok
        if has_ll_prefix(t):
            t_no_ll = drop_ll(t)
            key_ll = ar_normalize(t_no_ll)
            if key_ll in BUSINESS_MAP and BUSINESS_MAP[key_ll] not in {"Company","Office","Bank","Group","Holding"}:
                business_bits.append(BUSINESS_MAP[key_ll]); continue
            t = t_no_ll
        key = ar_normalize(t)
        if key in ORG_TYPE_TOKENS:
            org_suffix.append(BUSINESS_MAP.get(key, "Company")); continue
        if key in BUSINESS_MAP:
            eng = BUSINESS_MAP[key]
            if eng not in {"Company","Office","Bank","Group","Holding"}:
                business_bits.append(eng)
            continue
        if key in LEXICAL_MAP:
            en = LEXICAL_MAP[key]
            if has_al_prefix(t) and not en.lower().startswith("al-"):
                en = "Al-" + en
            body_en.append(en); continue
        if has_al_prefix(t):
            base = drop_al(t); en = translit_simple(base)
            body_en.append("Al-" + en)
        else:
            body_en.append(translit_simple(t))
    # de-dup labels
    labels, seen = [], set()
    for b in business_bits:
        if b not in seen:
            labels.append(b); seen.add(b)
    # suffix choice
    suffix_choice = None
    if org_suffix:
        for cand in ["Bank","Office","Group","Holding","Company"]:
            if cand in org_suffix: suffix_choice = cand; break
    if suffix_choice is None: suffix_choice = "Company"
    # compose suffix
    suffix = " ".join(labels + [suffix_choice]) if labels else suffix_choice
    # build final
    body_en = [w for w in body_en if w.lower() != "company"]
    result = cap_words(body_en).strip()
    if suffix: result = (result + " " + suffix).strip()
    result = re.sub(r"\s+", " ", result)
    result = re.sub(r"\b[Aa]l[- ]", "Al-", result)
    return result
