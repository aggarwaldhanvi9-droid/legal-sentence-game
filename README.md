# Legal Sentence Game

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ep8d2agpad/badge.svg)](https://ask.thili.ai/b/ep8d2agpad)

This project creates a Command-Line Interface (CLI) game designed for law students to practice drafting legal sentences. Players complete partial legal sentences using n-gram predictions, and their completed sentences are instantly classified by a fine-tuned GPT-2 model, providing valuable feedback on clause types.

## What you are building

**Goal.** Develop a CLI game where law students complete legal sentences using n-gram predictions, receiving instant feedback on generated clause types from a fine-tuned GPT-2 model.

**Done when.** The CLI game successfully presents partial legal sentences, allows players to complete them using n-gram predictions, and then classifies the full sentence with a fine-tuned GPT-2 model, providing feedback on the generated clause types.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a legal text corpus for n-gram model training.**  
      Hint: Use text preprocessing techniques from 'Build a Next-Word Predictor from Scratch' to clean and tokenize your corpus.
- [ ] **2. Implement the n-gram next-word predictor.**  
      Hint: Apply the n-gram probability calculation and prediction logic from 'Build a Next-Word Predictor from Scratch'.
- [ ] **3. Load the fine-tuned GPT-2 model for legal clause classification.**  
      Hint: Load your pre-trained GPT-2 model and tokenizer as demonstrated in 'Fine-Tune GPT-2 to Tag Legal Contract Clauses'.
- [ ] **4. Create a CLI game interface presenting partial legal sentences for completion.**  
      Hint: Focus on user input handling and displaying text in the console for the game flow.
- [ ] **5. After user completion, classify the full sentence and provide feedback.**  
      Hint: Use the fine-tuned GPT-2 model for inference on the completed sentence, similar to the classification step in 'Fine-Tune GPT-2 to Tag Legal Contract Clauses'.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Implement a scoring system based on the accuracy of the GPT-2 clause classification.
- Allow players to choose different difficulty levels, such as varying the length of the partial sentences provided.
- Track and display player statistics like average score or number of correct classifications over time.
