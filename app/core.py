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
        self.ngram_counts: dict[tuple[str, ...], Counter] =