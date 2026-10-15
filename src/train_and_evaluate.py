"""
Main Training and Evaluation Script

This script ties everything together:
1. Loads MIDI data from datasets
2. Splits into training and validation sets
3. Trains Markov chain models (first-order and higher-order)
4. Evaluates models using metrics like NLL
5. Generates new music samples

Usage:
    python train_and_evaluate.py --dataset nottingham --order 1 --max_files 100
"""

import os
import argparse
import numpy as np
from typing import List, Tuple

from midi_loader import load_dataset
from markov_chain import MarkovChain
from midi_generator import sequence_to_midi

def train_test_split(sequences: List[List], test_ratio: float = 0.2, 
                     random_seed: int = 42) -> Tuple[List[List], List[List]]:
    """
    Split sequences into training and validation sets.
    
    Parameters:
    -----------
    sequences : List[List]
        All sequences
    test_ratio : float
        Proportion of data to use for validation (0.0 to 1.0)
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
        (train_sequences, val_sequences)
    
    """
    np.random.seed(random_seed)
    
    # Shuffle sequences
    indices = np.random.permutation(len(sequences))
    split_idx = int(len(sequences) * (1 - test_ratio))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_sequences = [sequences[i] for i in train_indices]
    val_sequences = [sequences[i] for i in val_indices]
    
    return train_sequences, val_sequences


def evaluate_model(model: MarkovChain, val_sequences: List[List]) -> dict:
    """
    Evaluate a trained model on validation data.
    
    Parameters:
    -----------
    model : MarkovChain
        Trained model
    val_sequences : List[List]
        Validation sequences
    
    Returns:
    --------
        Dictionary of evaluation metrics
    """
    print("\nEvaluating model on validation set...")
    
    # Calculate NLL
    nll = model.calculate_negative_log_likelihood(val_sequences)
    
    # Calculate per-sequence log-likelihoods
    log_likelihoods = []
    for seq in val_sequences:
        if len(seq) >= model.order + 1:
            ll = model.calculate_log_likelihood(seq)
            if ll != float('-inf'):
                log_likelihoods.append(ll)
    
    avg_log_likelihood = np.mean(log_likelihoods) if log_likelihoods else float('-inf')
    
    # Calculate coverage (how many validation states were seen in training)
    val_states = set()
    for seq in val_sequences:
        val_states.update(seq)
    
    seen_states = model.all_states
    coverage = len(val_states & seen_states) / len(val_states) if val_states else 0.0
    
    metrics = {
        'nll': nll,
        'avg_log_likelihood': avg_log_likelihood,
        'coverage': coverage,
        'num_val_sequences': len(val_sequences),
        'num_valid_sequences': len(log_likelihoods)
    }
    
    return metrics


def print_metrics(metrics: dict, model_name: str = "Model"):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation Metrics")
    print(f"{'='*50}")
    print(f"Negative Log-Likelihood (NLL): {metrics['nll']:.4f}")
    print(f"Average Log-Likelihood: {metrics['avg_log_likelihood']:.4f}")
    print(f"State Coverage: {metrics['coverage']*100:.2f}%")
    print(f"Validation Sequences: {metrics['num_val_sequences']}")
    print(f"Valid Sequences (for evaluation): {metrics['num_valid_sequences']}")
    print(f"{'='*50}\n")


def main():
    """
    Main function that orchestrates the entire pipeline.
    
    Steps:
    1. Parse command-line arguments
    2. Load dataset
    3. Split into train/validation
    4. Train model(s)
    5. Evaluate model(s)
    6. Generate sample music
    """
    parser = argparse.ArgumentParser(description='Train and evaluate Markov chain music models')
    parser.add_argument('--dataset', type=str, default='nottingham',
                       choices=['nottingham', 'pop909'],
                       help='Dataset to use')
    parser.add_argument('--order', type=int, default=1,
                       help='Markov chain order (1=first-order, 2=second-order, etc.)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of MIDI files to load (None = all)')
    parser.add_argument('--include_rhythm', action='store_true',
                       help='Include rhythm (duration) in states')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Proportion of data for validation')
    parser.add_argument('--generate_samples', type=int, default=5,
                       help='Number of sample pieces to generate')
    parser.add_argument('--output_dir', type=str, default='../output',
                       help='Directory for generated MIDI files')
    
    args = parser.parse_args()
    
    # Determine dataset path
    if args.dataset == 'nottingham':
        data_dir = "data/nottingham_github/MIDI"
        track_name = None
    elif args.dataset == 'pop909':
        data_dir = "data/pop909/POP909"
        track_name = "MELODY"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print("="*60)
    print("Markov Chain Music Generation - Training Pipeline")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Order: {args.order}")
    print(f"Include Rhythm: {args.include_rhythm}")
    print(f"Max Files: {args.max_files or 'All'}")
    print("="*60)
    
    print("\n[Step 1] Loading dataset...")
    sequences = load_dataset(
        data_dir=data_dir,
        track_name=track_name,
        max_files=args.max_files,
        include_rhythm=args.include_rhythm
    )
    
    if not sequences:
        print("Error: No sequences loaded. Check dataset path.")
        return
    
    print(f"Loaded {len(sequences)} sequences")
    print(f"Average sequence length: {np.mean([len(s) for s in sequences]):.1f} states")
    
    print("\n[Step 2] Splitting into train/validation sets...")
    train_sequences, val_sequences = train_test_split(
        sequences, 
        test_ratio=args.test_ratio,
        random_seed=42
    )
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    
    print(f"\n[Step 3] Training {args.order}-order Markov chain...")
    model = MarkovChain(order=args.order)
    model.train(train_sequences)
    
    print("\n[Step 4] Evaluating model...")
    metrics = evaluate_model(model, val_sequences)
    print_metrics(metrics, f"{args.order}-order Markov Chain")
    
    print(f"\n[Step 5] Generating {args.generate_samples} sample pieces...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    generated_sequences = []
    for i in range(args.generate_samples):
        # Generate a sequence of similar length to training data
        avg_length = int(np.mean([len(s) for s in train_sequences]))
        generated = model.generate(length=avg_length, temperature=1.0)
        generated_sequences.append(generated)
        
        # Save individual MIDI file
        output_path = os.path.join(
            args.output_dir, 
            f"{args.dataset}_order{args.order}_sample{i+1}.mid"
        )
        sequence_to_midi(generated, output_path)
        print(f"  Generated sample {i+1}: {len(generated)} states -> {output_path}")
    
    # Save model
    model_path = os.path.join(args.output_dir, f"model_order{args.order}.pkl")
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    print("\n" + "="*60)
    print("Training and generation complete!")
    print("="*60)
    print(f"\nGenerated files are in: {args.output_dir}")
    print(f"Model file: {model_path}")
    print("\nYou can now:")
    print("  1. Listen to the generated MIDI files")
    print("  2. Compare different orders (run with --order 1, 2, 3)")
    print("  3. Compare with/without rhythm (--include_rhythm flag)")
    print("  4. Use the saved model to generate more music later")


if __name__ == "__main__":
    main()

