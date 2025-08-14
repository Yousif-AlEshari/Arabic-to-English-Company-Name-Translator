def ensure_splits_hf(raw):
    try:
        keys = set(raw.keys())
        if "validation" not in keys and "test" not in keys:
            tmp = raw["train"].train_test_split(test_size=0.2, seed=42)
            val_test = tmp["test"].train_test_split(test_size=0.5, seed=42)
            raw["train"], raw["validation"], raw["test"] = tmp["train"], val_test["train"], val_test["test"]
        elif "validation" not in keys:
            split = raw["train"].train_test_split(test_size=0.1, seed=42)
            raw["train"], raw["validation"] = split["train"], split["test"]
        elif "test" not in keys:
            split = raw["train"].train_test_split(test_size=0.1, seed=43)
            raw["train"], raw["test"] = split["train"], split["test"]
        return raw
    except Exception:
        return raw

def load_ar_en_splits():
    try:
        from datasets import load_dataset
        print("Loading IWSLT TED talks (2016) AR→EN …")
        raw = load_dataset("IWSLT/ted_talks_iwslt", language_pair=("ar", "en"), year="2016")
        raw = ensure_splits_hf(raw)
        return {"train": raw["train"], "validation": raw["validation"], "test": raw["test"]}
    except Exception as e:
        print(f"HF dataset unavailable ({e}). Using a tiny in-memory dataset for quick testing.")
        toy = [
            {"translation": {"ar": "مرحبا", "en": "hello"}},
            {"translation": {"ar": "كيف حالك", "en": "how are you"}},
            {"translation": {"ar": "أنا أحب البرمجة", "en": "i love programming"}},
            {"translation": {"ar": "هذا كتاب جديد", "en": "this is a new book"}},
            {"translation": {"ar": "السيارة سريعة جدا", "en": "the car is very fast"}},
            {"translation": {"ar": "السماء زرقاء اليوم", "en": "the sky is blue today"}},
            {"translation": {"ar": "رجل يقود دراجة على الطريق بجانب النهر", "en": "a man rides a bicycle on the road by the river"}},
        ]
        return {"train": toy[:5], "validation": toy[5:6], "test": toy[6:]}
