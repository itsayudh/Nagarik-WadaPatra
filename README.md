Nagarik WadaPatra

Welcome to the LLM assessment. This pack contains everything required to build a **small bilingual (EN/NP) domain LLM** for Nepal’s *Nagarik WadaPatra* 

Project objective 
Build a tiny, reproducible decoder-only Transformer (≤50M params) and a from-scratch SentencePiece 
tokenizer that can answer municipal citizen-charter questions (required documents, fees, processing time). 
Provide a CLI and a minimal FastAPI chat UI. Deliver bilingual support (English + Nepali Devanagari + 
Nepali numerals), deterministic training, Docker reproducibility.

Data plan  - Curate public municipal charter pages + official ward PDFs  - Create structured snippets and canonical fields (service name, required docs, fees, processing time, office). - Derive ≥600 instruction answer pairs (≥40% Nepali). Use templates and controlled randomization to 
augment.
