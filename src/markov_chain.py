"""
Markov Chain Model for Music Generation

Implements first-order and higher-order Markov chains with:
- Variable-order Katz/suffix backoff across orders k, k-1, ..., 1
- Additive Laplace smoothing
- Log-sum-exp numerically stable temperature sampling & greedy decoding (temperature <= 0)
- Per-transition negative log-likelihood (NLL), Perplexity, Pitch-Class Entropy,
  Diatonic Scale Adherence, and Harmonic Consonance scoring.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

MAJOR_SCALE_INTERVALS = {0, 2, 4, 5, 7, 9, 11}
NATURAL_MINOR_INTERVALS = {0, 2, 3, 5, 7, 8, 10}

# Semitone interval (mod 12) -> consonance weight in [0.1, 1.0]
INTERVAL_CONSONANCE: Dict[int, float] = {
    0: 1.00,   # Unison / Octave
    7: 0.95,   # Perfect 5th
    5: 0.90,   # Perfect 4th
    4: 0.88,   # Major 3rd
    3: 0.85,   # Minor 3rd
    9: 0.82,   # Major 6th
    8: 0.80,   # Minor 6th
    2: 0.55,   # Major 2nd
    10: 0.52,  # Minor 7th
    11: 0.40,  # Major 7th
    1: 0.25,   # Minor 2nd
    6: 0.20,   # Tritone
}


def extract_midi_pitch(state: Any) -> Optional[int]:
    """Extract an integer MIDI pitch from an int state or (pitch, duration) tuple."""
    if isinstance(state, (int, np.integer)):
        return int(state)
    if isinstance(state, tuple) and len(state) >= 1 and isinstance(state[0], (int, np.integer)):
        return int(state[0])
    return None


def score_harmonic_consonance(state_a: Any, state_b: Any) -> float:
    """Compute harmonic/melodic interval consonance in [0, 1] between two states."""
    pa = extract_midi_pitch(state_a)
    pb = extract_midi_pitch(state_b)
    if pa is None or pb is None:
        return 0.5
    semitones = abs(pb - pa)
    interval_mod = semitones % 12
    base = INTERVAL_CONSONANCE.get(interval_mod, 0.5)
    # Penalize extreme melodic leaps > 1 octave
    leap_penalty = max(0.0, (semitones - 12) * 0.03)
    return max(0.05, min(1.0, base - leap_penalty))


class MarkovChain:
    """
    Variable-order Markov chain model for musical sequence generation and evaluation.
    """

    def __init__(
        self,
        order: int = 1,
        smoothing_alpha: float = 0.0,
        enable_backoff: bool = True,
        harmony_weight: float = 0.0,
    ):
        if order < 1:
            raise ValueError("order must be >= 1")
        if smoothing_alpha < 0:
            raise ValueError("smoothing_alpha must be >= 0")

        self.order = int(order)
        self.smoothing_alpha = float(smoothing_alpha)
        self.enable_backoff = bool(enable_backoff)
        self.harmony_weight = float(harmony_weight)

        self.transition_matrix: Dict[Tuple, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.state_counts: Dict[Tuple, int] = defaultdict(int)
        # Lower-order suffix tables for variable-order backoff (order k-1 .. 1)
        self.backoff_matrices: Dict[int, Dict[Tuple, Dict[Any, int]]] = {
            k: defaultdict(lambda: defaultdict(int)) for k in range(1, self.order)
        }
        self.backoff_counts: Dict[int, Dict[Tuple, int]] = {
            k: defaultdict(int) for k in range(1, self.order)
        }
        self.unigram_counts: Dict[Any, int] = defaultdict(int)
        self.total_unigrams: int = 0
        self.all_states: set = set()

    def _get_state_sequence(self, sequence: Sequence, index: int, order: Optional[int] = None) -> Tuple:
        k = self.order if order is None else order
        start_idx = max(0, index - k)
        context = tuple(sequence[start_idx:index])
        while len(context) < k:
            context = (None,) + context
        return context

    def train(self, sequences: List[List]):
        """Train the Markov chain and all lower-order suffix backoff tables."""
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.state_counts = defaultdict(int)
        self.backoff_matrices = {
            k: defaultdict(lambda: defaultdict(int)) for k in range(1, self.order)
        }
        self.backoff_counts = {
            k: defaultdict(int) for k in range(1, self.order)
        }
        self.unigram_counts = defaultdict(int)
        self.total_unigrams = 0
        self.all_states = set()

        total_transitions = 0

        for sequence in sequences:
            if not sequence or len(sequence) < self.order + 1:
                continue

            for state in sequence:
                self.all_states.add(state)
                self.unigram_counts[state] += 1
                self.total_unigrams += 1

            for i in range(self.order, len(sequence)):
                context = self._get_state_sequence(sequence, i, self.order)
                next_state = sequence[i]
                self.transition_matrix[context][next_state] += 1
                self.state_counts[context] += 1
                total_transitions += 1

            for k in range(1, self.order):
                for i in range(k, len(sequence)):
                    sub_ctx = tuple(sequence[i - k : i])
                    next_state = sequence[i]
                    self.backoff_matrices[k][sub_ctx][next_state] += 1
                    self.backoff_counts[k][sub_ctx] += 1

    def _counts_to_distribution(
        self, counts_map: Dict[Any, int], total_count: int, last_state: Any = None
    ) -> Tuple[List, List[float]]:
        vocab = sorted(self.all_states, key=lambda x: str(x))
        if not vocab:
            return [], []

        if self.smoothing_alpha > 0.0:
            states = vocab
            denom = total_count + self.smoothing_alpha * len(vocab)
            probs = [(counts_map.get(s, 0) + self.smoothing_alpha) / denom for s in states]
        else:
            states = list(counts_map.keys())
            probs = [counts_map[s] / total_count for s in states]

        if self.harmony_weight > 0.0 and last_state is not None:
            weighted = []
            for s, p in zip(states, probs):
                cons = score_harmonic_consonance(last_state, s)
                factor = (1.0 - self.harmony_weight) + self.harmony_weight * cons
                weighted.append(p * factor)
            probs = weighted

        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        return states, probs

    def _get_transition_probabilities(self, context: Tuple) -> Tuple[List, List[float]]:
        """
        Get probability distribution for next state given context,
        backing off across suffix contexts of length k-1 .. 1 if unseen.
        """
        last_state = context[-1] if context else None

        if context in self.transition_matrix and self.state_counts[context] > 0:
            return self._counts_to_distribution(
                self.transition_matrix[context], self.state_counts[context], last_state
            )

        if self.enable_backoff and self.order > 1:
            for k in range(self.order - 1, 0, -1):
                sub_ctx = context[-k:]
                if sub_ctx in self.backoff_matrices[k] and self.backoff_counts[k][sub_ctx] > 0:
                    return self._counts_to_distribution(
                        self.backoff_matrices[k][sub_ctx],
                        self.backoff_counts[k][sub_ctx],
                        last_state,
                    )

            if self.total_unigrams > 0:
                return self._counts_to_distribution(
                    self.unigram_counts, self.total_unigrams, last_state
                )

        states = sorted(self.all_states, key=lambda x: str(x))
        if not states:
            return [], []
        probs = [1.0 / len(states)] * len(states)
        return states, probs

    def generate(
        self,
        length: int,
        start_context: Optional[Tuple] = None,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> List:
        """
        Generate a musical sequence of `length` states with numerically stable
        temperature scaling (and greedy decoding when temperature <= 0).
        """
        if not self.transition_matrix:
            raise ValueError("Model not trained yet. Call train() first.")
        if length <= 0:
            return []

        rng = np.random.default_rng(seed)
        generated = []

        if start_context is None:
            available_contexts = list(self.transition_matrix.keys())
            if not available_contexts:
                raise ValueError("No training data available")
            context = available_contexts[int(rng.integers(0, len(available_contexts)))]
        else:
            context = tuple(start_context)
            if len(context) < self.order:
                context = (None,) * (self.order - len(context)) + context
            elif len(context) > self.order:
                context = context[-self.order :]

        for _ in range(length):
            next_states, probabilities = self._get_transition_probabilities(context)

            if not next_states:
                all_states = sorted(self.all_states, key=lambda x: str(x))
                if not all_states:
                    break
                next_state = all_states[int(rng.integers(0, len(all_states)))]
            else:
                probs_arr = np.asarray(probabilities, dtype=np.float64)
                if temperature <= 1e-6:
                    idx = int(np.argmax(probs_arr))
                    next_state = next_states[idx]
                else:
                    if abs(temperature - 1.0) > 1e-9:
                        log_probs = np.log(np.maximum(probs_arr, 1e-12)) / float(temperature)
                        log_probs = log_probs - np.max(log_probs)
                        scaled = np.exp(log_probs)
                        probs_arr = scaled / np.sum(scaled)
                    else:
                        probs_arr = probs_arr / np.sum(probs_arr)
                    idx = int(rng.choice(len(next_states), p=probs_arr))
                    next_state = next_states[idx]

            generated.append(next_state)
            if self.order > 0:
                context = context[1:] + (next_state,)
            else:
                context = (next_state,)

        return generated

    def calculate_log_likelihood(self, sequence: List) -> float:
        """Calculate total log-likelihood of a sequence under this model."""
        if len(sequence) < self.order + 1:
            return float("-inf")

        log_likelihood = 0.0
        for i in range(self.order, len(sequence)):
            context = self._get_state_sequence(sequence, i)
            next_state = sequence[i]
            next_states, probabilities = self._get_transition_probabilities(context)

            if next_state in next_states:
                idx = next_states.index(next_state)
                prob = probabilities[idx]
                log_likelihood += float(np.log(max(prob, 1e-10)))
            else:
                log_likelihood += float(np.log(1e-10))

        return log_likelihood

    def calculate_negative_log_likelihood(self, sequences: List[List]) -> float:
        """Calculate average Negative Log-Likelihood (NLL) across sequences."""
        total_log_likelihood = 0.0
        valid_sequences = 0

        for sequence in sequences:
            if len(sequence) >= self.order + 1:
                ll = self.calculate_log_likelihood(sequence)
                if ll != float("-inf"):
                    total_log_likelihood += ll
                    valid_sequences += 1

        if valid_sequences == 0:
            return float("inf")

        return -(total_log_likelihood / valid_sequences)

    def calculate_perplexity(self, sequences: List[List]) -> float:
        """Calculate per-transition perplexity exp(-1/T * sum log P(x_t | context))."""
        total_ll = 0.0
        total_transitions = 0

        for seq in sequences:
            if len(seq) >= self.order + 1:
                ll = self.calculate_log_likelihood(seq)
                if math.isfinite(ll):
                    total_ll += ll
                    total_transitions += len(seq) - self.order

        if total_transitions == 0:
            return float("inf")

        avg_nll_per_token = -total_ll / total_transitions
        return float(math.exp(min(avg_nll_per_token, 50.0)))

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "order": self.order,
                    "smoothing_alpha": self.smoothing_alpha,
                    "enable_backoff": self.enable_backoff,
                    "harmony_weight": self.harmony_weight,
                    "transition_matrix": {
                        ctx: dict(counts) for ctx, counts in self.transition_matrix.items()
                    },
                    "state_counts": dict(self.state_counts),
                    "backoff_matrices": {
                        k: {ctx: dict(counts) for ctx, counts in mat.items()}
                        for k, mat in self.backoff_matrices.items()
                    },
                    "backoff_counts": {
                        k: dict(cnts) for k, cnts in self.backoff_counts.items()
                    },
                    "unigram_counts": dict(self.unigram_counts),
                    "total_unigrams": self.total_unigrams,
                    "all_states": list(self.all_states),
                },
                f,
            )

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.order = int(data["order"])
            self.smoothing_alpha = float(data.get("smoothing_alpha", 0.0))
            self.enable_backoff = bool(data.get("enable_backoff", True))
            self.harmony_weight = float(data.get("harmony_weight", 0.0))
            self.transition_matrix = defaultdict(
                lambda: defaultdict(int),
                {k: defaultdict(int, v) for k, v in data["transition_matrix"].items()},
            )
            self.state_counts = defaultdict(int, data["state_counts"])
            raw_backoff_mats = data.get("backoff_matrices", {})
            raw_backoff_cnts = data.get("backoff_counts", {})
            self.backoff_matrices = {
                k: defaultdict(
                    lambda: defaultdict(int),
                    {ctx: defaultdict(int, cnts) for ctx, cnts in raw_backoff_mats.get(k, {}).items()},
                )
                for k in range(1, self.order)
            }
            self.backoff_counts = {
                k: defaultdict(int, raw_backoff_cnts.get(k, {}))
                for k in range(1, self.order)
            }
            self.unigram_counts = defaultdict(int, data.get("unigram_counts", {}))
            self.total_unigrams = int(data.get("total_unigrams", 0))
            self.all_states = set(data["all_states"])


def evaluate_musical_sequence(sequence: Sequence[Any]) -> Dict[str, float]:
    """
    Compute music-theoretic and information-theoretic quality metrics for a generated sequence:
    - pitch_class_entropy_bits: Shannon entropy of pitch classes mod 12
    - max_diatonic_adherence: Best-fit major/minor key scale adherence in [0, 1]
    - mean_consonance: Mean melodic interval consonance in [0, 1]
    - stepwise_motion_ratio: Fraction of intervals <= 2 semitones
    """
    pitches = [p for p in (extract_midi_pitch(s) for s in sequence) if p is not None]
    if not pitches:
        return {
            "pitch_class_entropy_bits": 0.0,
            "max_diatonic_adherence": 0.0,
            "mean_consonance": 0.0,
            "stepwise_motion_ratio": 0.0,
        }

    pcs = [p % 12 for p in pitches]
    counts = Counter(pcs)
    n = len(pcs)
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())

    best_adherence = 0.0
    for tonic in range(12):
        maj_set = {(tonic + iv) % 12 for iv in MAJOR_SCALE_INTERVALS}
        min_set = {(tonic + iv) % 12 for iv in NATURAL_MINOR_INTERVALS}
        maj_score = sum(1 for pc in pcs if pc in maj_set) / n
        min_score = sum(1 for pc in pcs if pc in min_set) / n
        best_adherence = max(best_adherence, maj_score, min_score)

    if len(pitches) >= 2:
        consonances = [
            score_harmonic_consonance(pitches[i], pitches[i + 1])
            for i in range(len(pitches) - 1)
        ]
        intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
        mean_cons = sum(consonances) / len(consonances)
        stepwise = sum(1 for iv in intervals if iv <= 2) / len(intervals)
    else:
        mean_cons = 1.0
        stepwise = 1.0

    return {
        "pitch_class_entropy_bits": round(entropy, 4),
        "max_diatonic_adherence": round(best_adherence, 4),
        "mean_consonance": round(mean_cons, 4),
        "stepwise_motion_ratio": round(stepwise, 4),
    }
