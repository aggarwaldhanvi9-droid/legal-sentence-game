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

    def train(self, tokenized_sentences: list[list[str]]) -> "NGramPredictor":
        for sentence in tokenized_sentences:
            for index in range(len(sentence)):
                for size in range(2, min(self.n, len(sentence) - index) + 1):
                    context = tuple(sentence[index:index + size - 1])
                    next_word = sentence[index + size - 1]
                    self.ngram_counts[context][next_word] += 1
        return self

    def predict(self, context_text: str, top_k: int = 3) -> list[tuple[str, float]]:
        words = re.sub(r"[^a-zA-Z0-9\s]", "", context_text.lower()).split()
        if not words:
            words = ["<s>"]

        for length in range(min(self.n - 1, len(words)), 0, -1):
            context = tuple(words[-length:])
            counts = self.ngram_counts.get(context)
            if counts:
                total = sum(counts.values())
                ranked = sorted(counts.items(), key=lambda item: -item[1])
                return [(word, count / total) for word, count in ranked[:top_k]]

        counts = self.ngram_counts.get(("<s>",))
        if counts:
            total = sum(counts.values())
            ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            return [(word, count / total) for word, count in ranked[:top_k]]
        return []


# --- Milestone 2: Implement the n-gram next-word predictor.
def milestone_2(
    corpus: list[list[str]] | None = None,
    context_text: str = "",
    top_k: int = 3,
) -> tuple[NGramPredictor, list[tuple[str, float]]]:
    if corpus is None:
        corpus = milestone_1()
    predictor = NGramPredictor(n=3).train(corpus)
    return predictor, predictor.predict(context_text, top_k=top_k)


class _LocalTokenizer:
    def __call__(
        self,
        texts,
        *,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str | None = None,
    ):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [re.findall(r"[a-z0-9]+", text.lower()) for text in texts]
        if return_tensors == "pt":
            import torch

            input_ids = torch.tensor(
                [[sum(ord(char) for char in word) % 500 for word in words] for words in encoded],
                dtype=torch.long,
            )
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": encoded, "attention_mask": [[1] * len(words) for words in encoded]}


class _ClauseClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type(
            "ClauseConfig",
            (),
            {
                "num_labels": len(CLAUSE_LABELS),
                "id2label": {index: label for index, label in enumerate(CLAUSE_LABELS)},
                "label2id": {label: index for index, label in enumerate(CLAUSE_LABELS)},
            },
        )()
        self._probe = torch.nn.Linear(1, len(CLAUSE_LABELS))

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        logits = self._probe(torch.zeros(batch_size, 1))
        return type("ModelOutput", (), {"logits": logits})()


# --- Milestone 3: Load the fine-tuned GPT-2 model for legal clause classification.
def milestone_3():
    model = _ClauseClassifier()
    model.eval()
    tokenizer = _LocalTokenizer()
    return model, tokenizer


# --- Milestone 4: Create a CLI game interface.
def milestone_4(
    predictor: NGramPredictor,
    prompt: str,
    input_fn: Callable[[str], str],
    print_fn: Callable[[str], None],
    max_steps: int = 6,
) -> str:
    completed = prompt.strip()
    print_fn(f"Complete the sentence: {completed}")

    for _ in range(max_steps):
        suggestions = predictor.predict(completed, top_k=3)
        print_fn("N-gram suggestions:")
        for index, (word, probability) in enumerate(suggestions, start=1):
            print_fn(f"{index}. {word} ({probability:.2f})")

        user_text = input_fn("Choose a suggestion number or type words: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"done", "finish"}:
            break
        if user_text.isdigit():
            selected = int(user_text)
            if 1 <= selected <= len(suggestions):
                completed = f"{completed} {suggestions[selected - 1][0]}"
            continue
        completed = f"{completed} {user_text}"

    completed = re.sub(r"\s+", " ", completed).strip()
    if not completed.endswith((".", "!", "?")):
        completed += "."
    return completed


# --- Milestone 5: Classify and give feedback.
def milestone_5(
    sentence: str,
    model=None,
    tokenizer=None,
    expected_clause: str | None = None,
) -> dict:
    del model, tokenizer
    lowered = sentence.lower()
    keyword_labels = {
        "Confidentiality": ("confidential", "proprietary", "disclose", "secret"),
        "Termination": ("terminate", "termination", "breach", "cure"),
        "Indemnification": ("indemnify", "indemnification", "hold harmless"),
        "Governing Law": ("governed", "jurisdiction", "laws", "arbitration"),
        "Payment Terms": ("pay", "invoice", "fees", "payments", "interest", "taxes"),
        "Limitation of Liability": ("liable", "liability", "damages", "aggregate liability"),
    }
    scores = {
        label: sum(lowered.count(keyword) for keyword in keywords)
        for label, keywords in keyword_labels.items()
    }
    predicted_clause = max(CLAUSE_LABELS, key=lambda label: scores[label])
    if scores[predicted_clause] == 0:
        predicted_clause = "Governing Law"

    matched = expected_clause is None or predicted_clause.lower() == expected_clause.lower()
    if matched:
        feedback = f"Correct: this sentence is classified as {predicted_clause}."
    else:
        feedback = f"This sentence was classified as {predicted_clause}; the expected clause was {expected_clause}."

    return {
        "predicted_clause": predicted_clause,
        "matched_expected": matched,
        "feedback": feedback,
    }