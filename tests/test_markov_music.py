"""Unit tests for markov-music variable-order backoff, numerical stability, and harmony scoring."""

import pytest
from src.markov_chain import (
    MarkovChain,
    evaluate_musical_sequence,
    score_harmonic_consonance,
)
from src.cli import main as cli_main


CORPUS = [
    [60, 62, 64, 65, 67, 69, 71, 72],
    [60, 64, 67, 72, 67, 64, 60],
    [60, 62, 65, 67, 65, 64, 62, 60],
]


def test_variable_order_suffix_backoff_on_unseen_context():
    model = MarkovChain(order=3, enable_backoff=True)
    model.train(CORPUS)

    # Context (99, 69, 71) was never seen as a 3-gram, but 2-gram suffix (69, 71) -> 72 was seen!
    states, probs = model._get_transition_probabilities((99, 69, 71))
    assert states == [72]
    assert probs == [1.0]

    # Without backoff, unseen 3-gram falls back to uniform over all states
    no_backoff = MarkovChain(order=3, enable_backoff=False)
    no_backoff.train(CORPUS)
    u_states, u_probs = no_backoff._get_transition_probabilities((99, 69, 71))
    assert len(u_states) == len(no_backoff.all_states)
    assert abs(sum(u_probs) - 1.0) < 1e-9


def test_temperature_extremes_and_greedy_decoding():
    model = MarkovChain(order=1, smoothing_alpha=0.01)
    model.train(CORPUS)

    # Temperature = 0.0 (greedy decoding) and ultra-low temperature (0.0001) must not raise ZeroDivisionError or NaN
    seq_zero = model.generate(length=10, start_context=(60,), temperature=0.0, seed=7)
    seq_tiny = model.generate(length=10, start_context=(60,), temperature=0.0001, seed=7)
    assert len(seq_zero) == 10
    assert len(seq_tiny) == 10
    # After 60, state 62 has count 2 vs 64 with count 1, so both pick 62 first
    assert seq_zero[0] == 62
    assert seq_tiny[0] == 62

    # High temperature sampling
    seq_high = model.generate(length=12, temperature=2.5, seed=11)
    assert len(seq_high) == 12
    assert model.generate(length=0) == []


def test_perplexity_and_log_likelihood():
    model = MarkovChain(order=2, smoothing_alpha=0.1, enable_backoff=True)
    model.train(CORPUS)

    nll = model.calculate_negative_log_likelihood(CORPUS)
    ppl = model.calculate_perplexity(CORPUS)
    assert nll > 0
    assert 1.0 <= ppl < 20.0

    # Short sequence returns -inf log-likelihood
    assert model.calculate_log_likelihood([60]) == float("-inf")


def test_harmonic_consonance_and_musical_evaluation():
    # Perfect 5th (7 semitones) is much more consonant than tritone (6 semitones)
    assert score_harmonic_consonance(60, 67) > score_harmonic_consonance(60, 66)
    # Works on (pitch, duration) tuples too
    assert score_harmonic_consonance((60, 0.5), (67, 0.5)) > score_harmonic_consonance((60, 0.5), (61, 0.5))

    c_major_scale = [60, 62, 64, 65, 67, 69, 71, 72]
    eval_c = evaluate_musical_sequence(c_major_scale)
    assert eval_c["max_diatonic_adherence"] == 1.0
    assert eval_c["stepwise_motion_ratio"] == 1.0
    assert eval_c["pitch_class_entropy_bits"] > 2.5


def test_save_load_and_cli(tmp_path):
    model = MarkovChain(order=2, smoothing_alpha=0.05, harmony_weight=0.3)
    model.train(CORPUS)
    out_file = tmp_path / "model.pkl"
    model.save(str(out_file))

    loaded = MarkovChain(order=1)
    loaded.load(str(out_file))
    assert loaded.order == 2
    assert loaded.smoothing_alpha == 0.05
    assert loaded.harmony_weight == 0.3

    assert cli_main(["--order", "2", "--length", "12", "--temperature", "0.7"]) == 0
