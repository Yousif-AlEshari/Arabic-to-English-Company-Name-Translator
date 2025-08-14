# Company Name Translator

## Introduction

**Company Name Translator** is a neural machine translation (NMT) project designed to translate Arabic company names (and general sentences) into English. It leverages PyTorch for model training and inference, and provides both command-line and Streamlit web interfaces for easy interaction. The project is tailored for business and NLP practitioners who need accurate, domain-specific translations of company names and related text.

It is a practical Arabic→English translation system specialized for **company and organization names**. It blends:

- a **deterministic, rule-based renderer** that normalizes Arabic, maps domain/business tokens, and produces pronunciation-preserving transliterations (e.g., correct “Al-” handling), and  
- a **neural Seq2Seq (LSTM)** fallback for general sentences and out-of-domain inputs.

This hybrid routing yields clean, well-formed English names like `Al-Qasr Trading Company` instead of literal word-by-word renderings, while still being able to translate regular Arabic sentences.

---

## Purpose and Explanation

The main goal of this project is to provide a robust, customizable, and extensible translation tool for Arabic-to-English company names. Unlike generic translation APIs, this solution allows you to train, fine-tune, and deploy your own translation models using your proprietary datasets, ensuring privacy and domain adaptation. The project supports:

- Training a sequence-to-sequence (seq2seq) model on custom data.
- Translating new company names interactively or in batch.
- Evaluating translation quality using your own Excel files.
- Extending to other language pairs or domains with minimal changes.

---

## Features

- **Custom Dataset Support:** Train and evaluate on your own Excel files.
- **Modern NLP Pipeline:** Uses PyTorch, torchtext, and custom tokenizers.
- **Streamlit Web UI:** User-friendly interface for instant translation.
- **Artifact Management:** Save and load models, vocabularies, and configs.
- **Batch and Interactive Translation:** Translate single names or entire files.
- **Modular Codebase:** Easily extend or modify components (tokenizers, models, etc.).
- **Evaluation Tools:** Compare model output with ground truth in Excel.

---

## Project Structure and File Contents

### Folder: `Company_Name_Translator/`

| File/Folder         | Purpose                                                                                   |
|---------------------|-------------------------------------------------------------------------------------------|
| `__init__.py`       | Marks the directory as a Python package.                                                  |
| `app.py`            | Streamlit web application for interactive translation.                                    |
| `config.py`         | Configuration constants (paths, hyperparameters, etc.).                                   |
| `data_loading.py`   | Functions for loading and splitting datasets.                                             |
| `dataset.py`        | Custom PyTorch Dataset and DataLoader utilities for translation data.                     |
| `io_artifacts.py`   | Functions to save/load model artifacts (model, vocab, config).                            |
| `manual_input.xlsx` | Example Excel file for batch translation and evaluation.                                  |
| `model.py`          | Model definitions: Encoder, Decoder, Seq2Seq architecture.                                |
| `org_renderer.py`   | (Optional) Utilities for rendering organization names or results.                         |
| `routing.py`        | High-level translation logic and smart routing for inference.                             |
| `tokenizers.py`     | Tokenization functions for Arabic and English.                                            |
| `train.py`          | Main training and evaluation script (CLI).                                                |
| `translit.py`       | Utilities for transliteration (if needed).                                                |
| `vocab.py`          | Vocabulary building and management utilities.                                             |
| `__pycache__/`      | Compiled Python files for faster loading (auto-generated).                                |

### Folder: `artifacts_ar_en/`

| File                | Purpose                                                                                   |
|---------------------|-------------------------------------------------------------------------------------------|
| `config.json`       | Saved model configuration (hyperparameters, etc.).                                        |
| `model.pt`          | Trained PyTorch model weights.                                                            |
| `src_vocab.json`    | Source (Arabic) vocabulary mapping.                                                       |
| `trg_vocab.json`    | Target (English) vocabulary mapping.                                                      |

---

## Installation

1. **Clone the repository** (if not already done):

    ```powershell
    git clone <your-repo-url>
    cd "NLP Company name translator\GitHub Repo Download\Company_Name_Translator"
    ```

2. **Create and activate a virtual environment** (recommended):

    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```

3. **Install required packages:**

    ```powershell
    pip install torch torchtext pandas streamlit openpyxl
    ```

    *(Add any other dependencies as needed, e.g., `spacy`, `tqdm`.)*

---

## How to Run

### 1. **Train or Evaluate the Model (CLI)**

To train the model or run the demo from the command line:

```powershell
python train.py
```

- If a trained model exists in the artifacts directory, it will be loaded.
- If not, training will start using your dataset.

### 2. **Run the Streamlit Web App**

To launch the interactive web UI:

```powershell
streamlit run app.py
```

- Enter Arabic text in the input box and click "Translate" to see the English output.

---

## Notes

- **Custom Data:**  
  To use your own data, replace or edit `manual_input.xlsx` with your sentences. Update column names in the code if needed.

- **Artifacts:**  
  The `artifacts_ar_en` folder stores your trained model and vocabularies. Do not delete unless you want to retrain.

- **Python Version:**  
  For best compatibility, use Python 3.10 or 3.11. Some libraries may not fully support Python 3.12+.

- **GPU Support:**  
  If you have a CUDA-capable GPU and PyTorch installed with CUDA, training and inference will be faster.

- **Extending:**  
  You can adapt the code for other language pairs by updating tokenizers, vocab, and data loading logic.

- **Troubleshooting:**  
  If you encounter library or DLL errors, ensure all dependencies are installed in the same environment and are compatible.

- **Results:**
  The output of the model may not be 100% accurate, as this model was created for learning purposes, and was created with limited resources.
---

## Contact

For questions, suggestions, or contributions, please open an issue or pull request on the project repository.

---

## Author

Yousif Al Eshari
