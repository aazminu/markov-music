"""
Markov Chain Model for Music Generation

This module implements a Markov chain that can model musical sequences.
It supports both first-order (next state depends only on current state) and
higher-order (next state depends on multiple previous states) Markov chains.

Key Dependencies:
- MarkovChain: Main class for training and generating music
"""

import pickle
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional


class MarkovChain:
    """
    A Markov chain model for generating musical sequences.
    
    learns transition probabilities from training data and
    generate new sequences based on those probabilities.
    
    Members:
    -----------
    order : int
        The order of the Markov chain (1 = first-order, 2 = second-order, etc.)
    transition_matrix : dict
        Dictionary mapping state sequences to probability distributions
    state_counts : dict
        Dictionary counting occurrences of each state sequence
    all_states: set
        Set containing the states we've seen so far
    """
    
    def __init__(self, order: int = 1):
        """
        Initialize a Markov chain model.
        
        Parameters:
        -----------
        order : int
            The order of the Markov chain.
            - order=1: First-order (next state depends only on current state)
            - order=2: Second-order (next state depends on previous 2 states)
            - order=N: N-th order (next state depends on previous N states)
        """
        self.order = order
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.state_counts = defaultdict(int)
        self.all_states = set()
    
    def _get_state_sequence(self, sequence: List, index: int) -> Tuple:
        """
        Extract the state sequence (context) for prediction at a given index.
        
        Parameters:
        -----------
        sequence : List
            The full sequence of states
        index : int
            Current position in the sequence
        
        Returns:
        --------
            The state sequence (context) of length 'order'
        
        """
        # Get the previous 'order' states
        start_idx = max(0, index - self.order)
        context = tuple(sequence[start_idx:index])
        
        # Pad with None if we don't have enough history (for beginning of sequence)
        while len(context) < self.order:
            context = (None,) + context
        
        return context
    
    def train(self, sequences: List[List]):
        """
        Train the Markov chain on a list of sequences.
        
        Parameters:
        -----------
        sequences : List[List]
            List of training sequences. Each sequence is a list of states
            (e.g., [(pitch, duration), ...] or [pitch, ...])
        
        """
        print(f"Training {self.order}-order Markov chain on {len(sequences)} sequences...")
        
        # Reset the model
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.state_counts = defaultdict(int)
        self.all_states = set()
        
        total_transitions = 0
        
        # Process each sequence
        for seq_idx, sequence in enumerate(sequences):
            # Skip sequences that are too short
            if len(sequence) < self.order + 1:
                continue
            
            # Track all states we see
            for state in sequence:
                self.all_states.add(state)
            
            # Build transition counts
            for i in range(self.order, len(sequence)):
                context = self._get_state_sequence(sequence, i)
                
                next_state = sequence[i]
                
                # Increment the count for this transition
                self.transition_matrix[context][next_state] += 1
                self.state_counts[context] += 1
                total_transitions += 1
        
        print(f"Learned {len(self.transition_matrix)} unique contexts")
        print(f"Total transitions: {total_transitions}")
        print(f"Unique states: {len(self.all_states)}")
    
    def _get_transition_probabilities(self, context: Tuple) -> Tuple[List, List]:
        """
        Get the probability distribution for the next state given a context.
        
        Parameters:
        -----------
        context : Tuple
            The current context (previous N states)
        
        Returns:
        --------
        Tuple[List, List]
            (possible_next_states, probabilities) where probabilities sum to 1.0
        """
        if context not in self.transition_matrix:
            # If we've never seen this context, return uniform distribution
            # over all states we've seen
            states = list(self.all_states)
            if not states:
                return [], []
            probs = [1.0 / len(states)] * len(states)
            return states, probs
        
        next_state_counts = self.transition_matrix[context]
        total_count = self.state_counts[context]
        
        if total_count == 0:
            # Fallback: uniform distribution
            states = list(self.all_states)
            if not states:
                return [], []
            probs = [1.0 / len(states)] * len(states)
            return states, probs
        
        # Convert counts to probabilities
        states = []
        probs = []
        
        for state, count in next_state_counts.items():
            states.append(state)
            probs.append(count / total_count)
        
        # Normalize
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        
        return states, probs
    
    def generate(self, length: int, start_context: Optional[Tuple] = None, 
                 temperature: float = 1.0) -> List:
        """
        Generate a new sequence using the trained Markov chain.
        
        Parameters:
        -----------
        length : int
            Desired length of the generated sequence
        start_context : Tuple, optional
            Starting context (previous N states). If None, randomly selects
            a context that was seen during training.
        temperature : float
            Controls randomness (1.0 = normal, >1.0 = more random, <1.0 = more deterministic)
        
        Returns:
        --------
        List
            Generated sequence of states
        """
        if not self.transition_matrix:
            raise ValueError("Model not trained yet. Call train() first.")
        
        generated = []
        
        if start_context is None:
            # Randomly select a context that we've seen during training
            available_contexts = list(self.transition_matrix.keys())
            if not available_contexts:
                raise ValueError("No training data available")
            context = available_contexts[np.random.randint(len(available_contexts))]
        else:
            context = start_context
        
        # Generate sequence
        for _ in range(length):
            # Get probability distribution for next state
            next_states, probabilities = self._get_transition_probabilities(context)
            
            if not next_states:
                # If no transitions available, pick a random state
                all_states = list(self.all_states)
                if not all_states:
                    break
                next_state = all_states[np.random.randint(len(all_states))]
            else:
                # Apply temperature to probabilities
                if temperature != 1.0:
                    # Convert to log space, divide by temperature, convert back
                    log_probs = np.log(np.array(probabilities) + 1e-10)
                    scaled_log_probs = log_probs / temperature
                    probabilities = np.exp(scaled_log_probs)
                    probabilities = probabilities / probabilities.sum()
                
                # Sample from the distribution
                # Use index-based sampling to handle tuples correctly
                idx = np.random.choice(len(next_states), p=probabilities)
                next_state = next_states[idx]
            
            generated.append(next_state)
            
            # shift and add new state
            if self.order > 0:
                context = context[1:] + (next_state,)
            else:
                context = (next_state,)
        
        return generated
    
    def calculate_log_likelihood(self, sequence: List) -> float:
        """
        Calculate the log-likelihood of a sequence under this model.
        
        Parameters:
        -----------
        sequence : List
            A sequence of states to evaluate
        
        Returns:
        --------
        float
            Log-likelihood of the sequence (higher is better)
        """
        if len(sequence) < self.order + 1:
            return float('-inf')
        
        log_likelihood = 0.0
        
        for i in range(self.order, len(sequence)):
            context = self._get_state_sequence(sequence, i)
            next_state = sequence[i]
            
            # Get probability of this transition
            next_states, probabilities = self._get_transition_probabilities(context)
            
            if next_state in next_states:
                idx = next_states.index(next_state)
                prob = probabilities[idx]
                # Add log probability (use small epsilon to avoid log(0))
                log_likelihood += np.log(prob + 1e-10)
            else:
                # State never seen in this context
                log_likelihood += np.log(1e-10)
        
        return log_likelihood
    
    def calculate_negative_log_likelihood(self, sequences: List[List]) -> float:
        """
        Calculate the average Negative Log-Likelihood (NLL) on a set of sequences.
        
        Parameters:
        -----------
        sequences : List[List]
            List of sequences to evaluate
        
        Returns:
        --------
        float
            Average NLL (lower is better)
        """
        total_log_likelihood = 0.0
        valid_sequences = 0
        
        for sequence in sequences:
            if len(sequence) >= self.order + 1:
                ll = self.calculate_log_likelihood(sequence)
                if ll != float('-inf'):
                    total_log_likelihood += ll
                    valid_sequences += 1
        
        if valid_sequences == 0:
            return float('inf')
        
        avg_log_likelihood = total_log_likelihood / valid_sequences
        return -avg_log_likelihood  # Return negative (NLL)
    
    def save(self, filepath: str):
        """Save the trained model to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'order': self.order,
                'transition_matrix': dict(self.transition_matrix),
                'state_counts': dict(self.state_counts),
                'all_states': list(self.all_states)
            }, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.order = data['order']
            self.transition_matrix = defaultdict(lambda: defaultdict(int), 
                                                 {k: dict(v) for k, v in data['transition_matrix'].items()})
            self.state_counts = defaultdict(int, data['state_counts'])
            self.all_states = set(data['all_states'])
        print(f"Model loaded from {filepath}")

