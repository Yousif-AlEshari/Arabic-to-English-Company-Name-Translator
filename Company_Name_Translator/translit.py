import re

SUN_LETTERS = set("تثدذرزسشصضطظلن")
AR_DIACRITICS = ''.join([
    '\u0610','\u0611','\u0612','\u0613','\u0614','\u0615','\u0616','\u0617','\u0618','\u0619','\u061A',
    '\u064B','\u064C','\u064D','\u064E','\u064F','\u0650','\u0651','\u0652','\u0653','\u0654','\u0655',
    '\u0656','\u0657','\u0658','\u0659','\u065A','\u065B','\u065C','\u065D','\u065E','\u065F','\u0670'
])
TATWEEL = '\u0640'

def ar_normalize(s: str) -> str:
    s = s.replace('أ','ا').replace('إ','ا').replace('آ','ا')
    s = s.replace('ى','ي').replace(TATWEEL, '')
    s = re.sub(f"[{AR_DIACRITICS}]", "", s)
    return re.sub(r"\s+", " ", s).strip()

BASE = {
    'ا':'a','ب':'b','ت':'t','ث':'th','ج':'j','ح':'h','خ':'kh','د':'d','ذ':'dh','ر':'r','ز':'z',
    'س':'s','ش':'sh','ص':'s','ض':'d','ط':'t','ظ':'z','ع':"a", 'غ':'gh','ف':'f','ق':'q','ك':'k',
    'ل':'l','م':'m','ن':'n','ه':'h','و':'w','ي':'y','ء':"'", 'ؤ':"'", 'ئ':"'", 'ة':'t',
}

def transliterate_token(tok: str,
                        assimilate_al: bool = True,
                        final_ta_marbuta_a: bool = True,
                        double_for_shadda: bool = True) -> str:
    if not tok: return tok
    out = ""
    if double_for_shadda and '\u0651' in tok:
        tok = re.sub(r'([ءاأإآبتثجحخدذرزسشصضطظعغفقكلمنهوىي]\u0651)',
                     lambda m: m.group(0)[0]*2, tok).replace('\u0651','')
    end_as_a = False
    if final_ta_marbuta_a and tok.endswith('ة'):
        tok = tok[:-1] + 'ه'
        end_as_a = True
    if tok.startswith('ال') and len(tok) >= 2:
        nxt = tok[2:3]
        if assimilate_al and nxt and nxt in SUN_LETTERS:
            cons = transliterate_token(nxt, assimilate_al=False, final_ta_marbuta_a=False, double_for_shadda=False)
            out = "a" + cons[0] + "-"
            tok = tok[2:]
        else:
            out = "al-"; tok = tok[2:]
    for ch in tok:
        out += BASE.get(ch, ch)
    if end_as_a and (out.endswith('h') or out.endswith('t')):
        out = out[:-1] + 'a'
    out = re.sub(r"\'(?=[aeiou])", "", out)
    out = re.sub(r"\-+", "-", out).strip('-')
    return out

def transliterate_arabic_name(s: str) -> str:
    s = ar_normalize(s)
    return " ".join(transliterate_token(p) for p in s.split())
