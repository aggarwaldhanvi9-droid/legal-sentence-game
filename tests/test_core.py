"""Tests for Milestones 1 through 5 of Legal Sentence Game."""
from app.core import (
    CLAUSE_LABELS,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
)


# Milestone 1: Prepare a legal text corpus for n-gram model training.
def test_milestone_1():
    tokenized = milestone_1(["The Party shall pay!", "All data remains confidential."])
    assert len(tokenized) == 2
    assert tokenized[0] == ["<s>", "the", "party", "shall", "pay", "</s>"]
    assert tokenized[1] == ["<s>", "all", "data", "remains", "confidential", "</s>"]


# Milestone 2: Implement the n-gram next-word predictor.
def test_milestone_2():
    corpus = milestone_1()
    predictor, predictions = milestone_2(corpus, context_text="the receiving party", top_k=3)
    assert len(predictions) >= 1
    top_word, prob = predictions[0]
    assert top_word == "shall"
    assert 0.0 < prob <= 1.0


# Milestone 3: Load the fine-tuned GPT-2 model for legal clause classification.
def test_milestone_3():
    model, tokenizer = milestone_3()
    assert model is not None
    assert tokenizer is not None
    assert model.config.num_labels == len(CLAUSE_LABELS)
    assert model.config.id2label[0] == "Confidentiality"


# Milestone 4: Create a CLI game interface presenting partial legal sentences for completion.
def test_milestone_4():
    predictor, _ = milestone_2()
    inputs = iter(["1", "strictly secret", "done"])
    printed: list[str] = []

    completed = milestone_4(
        predictor=predictor,
        prompt="the receiving party shall",
        input_fn=lambda _: next(inputs),
        print_fn=printed.append,
        max_steps=4,
    )
    assert completed.startswith("the receiving party shall keep")
    assert completed.endswith(".")
    assert any("N-gram suggestions:" in line for line in printed)


# Milestone 5: After user completion, classify the full sentence and provide feedback.
def test_milestone_5():
    model, tokenizer = milestone_3()
    sentence = "either party may terminate this agreement upon thirty days prior written notice."
    result = milestone_5(
        sentence,
        model=model,
        tokenizer=tokenizer,
        expected_clause="Termination",
    )
    assert result["predicted_clause"] == "Termination"
    assert result["matched_expected"] is True
    assert "Termination" in result["feedback"]