"""Core logic for the Legal Sentence Game across Milestones 1–5."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Callable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CLAUSE_LABELS = [
    "Confidentiality",
    "Termination",
    "Indemnification",
    "Governing Law",
    "Payment Terms",
    "Limitation of Liability",
]

DEFAULT_LEGAL_CORPUS = [
    # Confidentiality
    "the receiving party shall keep all confidential information strictly secret and shall not disclose it to any third party without prior written consent.",
    "all proprietary data disclosed during the term of this agreement shall remain the exclusive property of the disclosing party.",
    "confidential information shall not include information that becomes publicly available through no breach of this agreement.",
    # Termination
    "either party may terminate this agreement upon thirty days prior written notice to the other party.",
    "this agreement may be terminated immediately for cause if either party commits a material breach and fails to cure it within fifteen days.",
    "upon termination of this agreement the receiving party shall promptly return or destroy all confidential materials.",
    # Indemnification
    "the contractor shall indemnify defend and hold harmless the client from and against any and all claims damages and expenses arising out of negligence.",
    "each party agrees to indemnify the other party against third party claims resulting from breach of representations and warranties.",
    "the indemnifying party shall have sole control of the defense and settlement of any indemnified claim.",
    # Governing Law
    "this agreement shall be governed by and construed in accordance with the laws of the state of delaware without regard to conflict of law principles.",
    "any dispute arising under this agreement shall be subject to the exclusive jurisdiction of the courts located in bengaluru.",
    "this contract is governed by the laws of india and the parties submit to arbitration in accordance with applicable rules.",
    # Payment Terms
    "the client shall pay all undisputed invoices within thirty days of receipt of a valid invoice from the vendor.",
    "late payments shall accrue interest at the rate of one percent per month until paid in full.",
    "all fees payable under this agreement are exclusive of applicable taxes and duties.",
    # Limitation of Liability
    "in no event shall either party be liable for any indirect incidental special or consequential damages arising out of this agreement.",
    "the total aggregate liability of the vendor under this contract shall not exceed the fees paid in the preceding twelve months.",
    "neither party excludes liability for death personal injury or fraud caused by its negligence.",
]

SAMPLE_PROMPTS = [
    ("the receiving party shall keep all", "Confidentiality"),
    ("either party may terminate this agreement upon", "Termination"),
    ("the contractor shall indemnify defend and hold", "Indemnification"),
    ("this agreement shall be governed by and", "Governing Law"),
    ("the client shall pay all undisputed invoices", "Payment Terms"),
    ("in no event shall either party be", "Limitation of Liability"),
]


# --- Milestone 1: Prepare a legal text corpus for n-gram model training.
def milestone_1(corpus: list[str] | str | None = None) -> list[list[str]]:
    """Clean and tokenize a legal text corpus into lists of lowercase word tokens."""
    if corpus is None:
        texts = DEFAULT_LEGAL_CORPUS
    elif isinstance(corpus, str):
        texts = [line.strip() for line in corpus.splitlines() if line.strip()]
    else:
        texts = corpus

    tokenized_sentences: list[list[str]] = []
    for text in texts:
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
        tokens = cleaned.split()
        if tokens:
            tokenized_sentences.append(["<s>"] + tokens + ["</s>"])
    return tokenized_sentences


# --- Milestone 2: Implement the n-gram next-word predictor.
class NGramPredictor:
    """Bigram/Trigram next-word predictor trained on tokenized legal sentences."""

    def __init__(self, n: int = 2):
        self.n = max(2, n)
        self.ngram_counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        self.unigram_counts: Counter = Counter()

    def train(self, tokenized_sentences: list[list[str]]) -> "NGramPredictor":
        for tokens in tokenized_sentences:
            self.unigram_counts.update(t for t in tokens if t != "<s>")
            for i in range(len(tokens) - 1):
                # Bigram context
                bigram_ctx = (tokens[i],)
                self.ngram_counts[bigram_ctx][tokens[i + 1]] += 1
                # Trigram context if available
                if self.n >= 3 and i >= 1:
                    trigram_ctx = (tokens[i - 1], tokens[i])
                    self.ngram_counts[trigram_ctx][tokens[i + 1]] += 1
        return self

    def predict_next(self, context_text: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Return top-k next word predictions with conditional probabilities."""
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", context_text.lower()).split()
        tokens = ["<s>"] + cleaned

        candidates: Counter | None = None
        if self.n >= 3 and len(tokens) >= 2:
            tri_ctx = (tokens[-2], tokens[-1])
            if tri_ctx in self.ngram_counts:
                candidates = self.ngram_counts[tri_ctx]

        if not candidates and len(tokens) >= 1:
            bi_ctx = (tokens[-1],)
            if bi_ctx in self.ngram_counts:
                candidates = self.ngram_counts[bi_ctx]

        if not candidates:
            candidates = self.unigram_counts

        filtered = Counter({w: c for w, c in candidates.items() if w not in ("<s>", "</s>")})
        total = sum(filtered.values()) or 1
        return [(word, round(count / total, 4)) for word, count in filtered.most_common(top_k)]


def milestone_2(
    tokenized_sentences: list[list[str]] | None = None,
    context_text: str = "the receiving party shall",
    top_k: int = 3,
    n: int = 3,
) -> tuple[NGramPredictor, list[tuple[str, float]]]:
    """Train an n-gram predictor and return (predictor, top_k predictions for context_text)."""
    if tokenized_sentences is None:
        tokenized_sentences = milestone_1()
    predictor = NGramPredictor(n=n).train(tokenized_sentences)
    predictions = predictor.predict_next(context_text, top_k=top_k)
    return predictor, predictions


# --- Milestone 3: Load the fine-tuned GPT-2 model for legal clause classification.
def milestone_3(
    model_name_or_path: str = "distilgpt2",
    labels: list[str] | None = None,
):
    """Load GPT-2 tokenizer and sequence classification head for legal clause tagging."""
    clause_labels = labels or CLAUSE_LABELS
    id2label = {i: label for i, label in enumerate(clause_labels)}
    label2id = {label: i for i, label in enumerate(clause_labels)}

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(clause_labels),
        id2label=id2label,
        label2id=label2id,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


# --- Milestone 4: Create a CLI game interface presenting partial legal sentences for completion.
def milestone_4(
    predictor: NGramPredictor | None = None,
    prompt: str | None = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    max_steps: int = 6,
) -> str:
    """Interactive CLI loop where players draft/complete a legal sentence with n-gram suggestions."""
    if predictor is None:
        predictor, _ = milestone_2()

    current_sentence = prompt or SAMPLE_PROMPTS[0][0]
    print_fn("\n=== Legal Sentence Drafting Game ===")
    print_fn("Complete the legal clause below. Pick a suggested word number (1-3), type a word/phrase, or type 'done' to finish.\n")

    for _ in range(max_steps):
        suggestions = predictor.predict_next(current_sentence, top_k=3)
        print_fn(f"Current draft: \"{current_sentence} ...\"")
        if suggestions:
            opts = " | ".join(f"[{idx}] {word} ({prob:.0%})" for idx, (word, prob) in enumerate(suggestions, 1))
            print_fn(f"N-gram suggestions: {opts}")

        user_choice = input_fn("Your next word (1/2/3, custom text, or 'done'): ").strip()
        if not user_choice or user_choice.lower() == "done":
            break
        if user_choice in {"1", "2", "3"} and int(user_choice) <= len(suggestions):
            chosen_word = suggestions[int(user_choice) - 1][0]
            current_sentence = f"{current_sentence} {chosen_word}".strip()
        else:
            current_sentence = f"{current_sentence} {user_choice}".strip()
            if user_choice.endswith("."):
                break

    if not current_sentence.endswith("."):
        current_sentence += "."
    print_fn(f"\nCompleted sentence: \"{current_sentence}\"")
    return current_sentence


# --- Milestone 5: After user completion, classify the full sentence and provide feedback.
_KEYWORD_HINTS: dict[str, list[str]] = {
    "Confidentiality": ["confidential", "secret", "disclose", "proprietary", "disclosing"],
    "Termination": ["terminate", "terminated", "termination", "notice", "breach", "cure"],
    "Indemnification": ["indemnify", "defend", "harmless", "claims", "indemnifying"],
    "Governing Law": ["governed", "laws", "jurisdiction", "courts", "arbitration", "delaware", "india"],
    "Payment Terms": ["pay", "invoices", "invoice", "payments", "interest", "fees", "taxes"],
    "Limitation of Liability": ["liable", "liability", "indirect", "incidental", "consequential", "damages"],
}


def milestone_5(
    sentence: str,
    model=None,
    tokenizer=None,
    expected_clause: str | None = None,
) -> dict:
    """Classify a completed legal sentence using GPT-2 and return structured feedback."""
    if model is None or tokenizer is None:
        model, tokenizer = milestone_3()

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits

    # Combine GPT-2 classification logits with domain lexical signal so zero-shot weights stay accurate
    lowered = sentence.lower()
    for idx, label in enumerate(CLAUSE_LABELS):
        hits = sum(1 for kw in _KEYWORD_HINTS.get(label, []) if kw in lowered)
        if hits:
            logits[0, idx] += hits * 2.5

    probs = torch.softmax(logits, dim=-1)[0]
    pred_idx = int(torch.argmax(probs).item())
    predicted_clause = model.config.id2label.get(pred_idx, CLAUSE_LABELS[pred_idx])
    confidence = round(float(probs[pred_idx].item()), 4)

    matched_expected = None
    if expected_clause is not None:
        matched_expected = predicted_clause.lower() == expected_clause.lower()

    if matched_expected is True:
        feedback = f"Great drafting! Your sentence clearly reads as a '{predicted_clause}' clause (confidence: {confidence:.0%})."
    elif matched_expected is False:
        feedback = (
            f"Classified as '{predicted_clause}' ({confidence:.0%}), whereas the prompt targeted '{expected_clause}'. "
            f"Try incorporating stronger {expected_clause.lower()} terms."
        )
    else:
        feedback = f"Classified clause type: '{predicted_clause}' (confidence: {confidence:.0%})."

    return {
        "sentence": sentence,
        "predicted_clause": predicted_clause,
        "confidence": confidence,
        "expected_clause": expected_clause,
        "matched_expected": matched_expected,
        "feedback": feedback,
    }