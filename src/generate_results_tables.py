"""
Generate Results Tables from Experiment Outputs

This script parses the output from train_and_evaluate.py and creates
formatted tables for the paper's results section.

"""

import os

from typing import Dict, List

def parse_experiment_output(output_dir: str) -> Dict[str, float]:
    """
    Parse experiment results from a directory.
    Looks for the printed metrics in the console output or saved files.
    """
    # Check if there's a model file (indicates experiment ran)
    model_file = None
    for order in [1, 3, 5]:
        potential_file = os.path.join(output_dir, f"model_order{order}.pkl")
        if os.path.exists(potential_file):
            model_file = potential_file
            break
    
    if not model_file:
        return None
    
    # Extract info from directory name
    parts = output_dir.split('/')
    dataset = parts[-2] if len(parts) > 1 else "unknown"
    config = parts[-1] if parts else "unknown"
    
    # Parse configuration
    if "order1" in config:
        order = 1
    elif "order3" in config:
        order = 3
    elif "order5" in config:
        order = 5
    else:
        order = 1
    has_rhythm = "with_rhythm" in config
    
    return {
        'dataset': dataset,
        'order': order,
        'has_rhythm': has_rhythm,
        'config': config,
        'output_dir': output_dir
    }


def create_results_table(experiments: List[Dict]) -> str:
    """Create a LaTeX table of results."""
    
    table = """\\begin{table}[h]
\\centering
\\caption{Model Performance Metrics}
\\label{tab:model_performance}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Configuration} & \\textbf{NLL} & \\textbf{Avg Log-Likelihood} & \\textbf{Coverage} & \\textbf{Unique States} \\\\
\\midrule
"""
    
    for exp in experiments:
        config_name = f"{exp['dataset'].capitalize()}, Order {exp['order']}"
        if exp['has_rhythm']:
            config_name += ", With Rhythm"
        else:
            config_name += ", Pitch Only"
        
        # Placeholder values - replace with actual results
        table += f"{config_name} & XXX.XX & -XXX.XX & XX.XX\\% & XXXX \\\\\n"
    
    table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return table


def create_markdown_table(experiments: List[Dict]) -> str:
    """Create a Markdown table of results."""
    
    table = "## Model Performance Metrics\n\n"
    table += "| Configuration | NLL | Avg Log-Likelihood | Coverage | Unique States |\n"
    table += "|---------------|-----|---------------------|----------|---------------|\n"
    
    for exp in experiments:
        config_name = f"{exp['dataset'].capitalize()}, Order {exp['order']}"
        if exp['has_rhythm']:
            config_name += ", With Rhythm"
        else:
            config_name += ", Pitch Only"
        
        # Placeholder values - replace with actual results
        table += f"| {config_name} | XXX.XX | -XXX.XX | XX.XX% | XXXX |\n"
    
    return table


def main():
    """Main function to generate tables."""
    
    results_dir = "results"
    
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} directory not found.")
        print("Please run experiments first using run_experiments.sh")
        return
    
    # Find all experiment directories
    experiments = []
    
    for dataset in ['nottingham', 'pop909']:
        dataset_dir = os.path.join(results_dir, dataset)
        if not os.path.exists(dataset_dir):
            continue
        
        for config_dir in os.listdir(dataset_dir):
            full_path = os.path.join(dataset_dir, config_dir)
            if os.path.isdir(full_path):
                exp_info = parse_experiment_output(full_path)
                if exp_info:
                    experiments.append(exp_info)
    
    if not experiments:
        print("No experiments found. Please run experiments first.")
        return
    
    # Sort experiments
    experiments.sort(key=lambda x: (x['dataset'], x['order'], x['has_rhythm']))
    
    # Generate tables
    print("="*60)
    print("GENERATED TABLES")
    print("="*60)
    print("\n--- LaTeX Table ---\n")
    print(create_results_table(experiments))
    print("\n--- Markdown Table ---\n")
    print(create_markdown_table(experiments))
    
    # Save to file
    with open("results/results_tables.txt", "w") as f:
        f.write("LATEX TABLE:\n")
        f.write("="*60 + "\n")
        f.write(create_results_table(experiments))
        f.write("\n\nMARKDOWN TABLE:\n")
        f.write("="*60 + "\n")
        f.write(create_markdown_table(experiments))
    
    print("\nTables saved to results/results_tables.txt")

if __name__ == "__main__":
    main()

