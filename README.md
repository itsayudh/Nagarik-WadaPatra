Nagarik WadaPatra

Welcome to the LLM assessment. This pack contains everything required to build a **small bilingual (EN/NP) domain LLM** for Nepal’s *Nagarik WadaPatra* 

Project objective 
Build a tiny, reproducible decoder-only Transformer (≤50M params) and a from-scratch SentencePiece 
tokenizer that can answer municipal citizen-charter questions (required documents, fees, processing time). 
Provide a CLI and a minimal FastAPI chat UI. Deliver bilingual support (English + Nepali Devanagari + 
Nepali numerals), deterministic training, Docker reproducibility.

Data plan  - Curate public municipal charter pages + official ward PDFs  - Create structured snippets and canonical fields (service name, required docs, fees, processing time, office). - Derive 200+ instruction answer pairs (≥40% Nepali). Use templates and controlled randomization to 
augment.

## Features

- ✅ Extracts and cleans text from multi-page PDFs
- ✅ Formats multilingual Q&A pairs (≥40% Nepali)
- ✅ Trains a subword tokenizer using SentencePiece
- ✅ Outputs ready-to-ingest files for chatbot and NLP pipelines

## 🛠 Requirements

- Python 3.7+
- PyPDF2
- sentencepiece
- 
 Extract Text from PDFs
 python extract_text.py


Clean the Corpus
python clean_corpus.py

Generate Q&A Pairs
python format_qa_pairs.py

Train Tokenizer
python tokenizer_train.py

Notes
- Tokenizer supports both unigram and bpe model types
- Corpus cleaning ensures consistent formatting for downstream tasks
- Q&A pairs are manually curated for accuracy and diversity

