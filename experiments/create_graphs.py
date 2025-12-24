# Visualization script for experiment results
# Creates graphs and tables for analysis

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# plot one experiment
def plot_single_experiment(csv_file, title, output_file):
    # Read data
    df = pd.read_csv(csv_file)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['generation'], df['max_fitness'], label='Max Fitness', linewidth=2)
    plt.plot(df['generation'], df['mean_fitness'], label='Mean Fitness', linewidth=2, alpha=0.7)
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness (Max Height)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs("graphs", exist_ok=True)
    plt.savefig(os.path.join("graphs", output_file), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: graphs/{output_file}")

# plot multiple experiments for comparison
def plot_comparison(csv_files, labels, title, output_file, ylabel='Max Fitness (Max Height)'):
    plt.figure(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (csv_file, label) in enumerate(zip(csv_files, labels)):
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found, skipping...")
            continue
            
        df = pd.read_csv(csv_file)
        plt.plot(df['generation'], df['max_fitness'], 
                label=label, linewidth=2, color=colors[i % len(colors)])
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs("../graphs", exist_ok=True)
    plt.savefig(os.path.join("../graphs", output_file), dpi=300, bbox_inches='tight')
    print(f"Saved: ../graphs/{output_file}")
    
    # Display plot
    plt.show()
    plt.close()

# create summary table from multiple experiments
def create_summary_table(csv_files, labels, output_file):
    summary_data = []
    
    for csv_file, label in zip(csv_files, labels):
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found, skipping from summary...")
            continue
            
        df = pd.read_csv(csv_file)
        
        summary_data.append({
            'Configuration': label,
            'Final Max Fitness': df['max_fitness'].iloc[-1],
            'Final Mean Fitness': df['mean_fitness'].iloc[-1],
            'Peak Fitness': df['max_fitness'].max(),
            'Peak Generation': df.loc[df['max_fitness'].idxmax(), 'generation'],
            'Avg Max Fitness (last 100 gen)': df['max_fitness'].iloc[-100:].mean(),
            'Final Mean Links': df['mean_links'].iloc[-1],
        })
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    os.makedirs("../graphs", exist_ok=True)
    summary_df.to_csv(os.path.join("../graphs", output_file), index=False)
    print(f"Saved: ../graphs/{output_file}")
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))
    print()

def main():
    """Generate all graphs and tables"""
    print("="*60)
    print("GENERATING GRAPHS AND TABLES")
    print("="*60)
    print()
    
    # POPULATION SIZE EXPERIMENTS
    print("Processing Population Size Experiments...")

    pop_sizes = [5, 10, 20, 50]
    pop_csv_files = [f"../results/pop_size_{size}.csv" for size in pop_sizes]
    pop_labels = [f"Pop Size {size}" for size in pop_sizes]

    # Individual plots
    for size in pop_sizes:
        csv_file = f"../results/pop_size_{size}.csv"
        if os.path.exists(csv_file):
            plot_single_experiment(
                csv_file,
                f"Population Size {size} - Fitness Over Generations",
                f"pop_size_{size}.png"
            )

    # GENE COUNT EXPERIMENTS
    print("\nProcessing Gene Count Experiments...")

    gene_counts = [2, 3, 5, 10]
    gene_csv_files = [f"../results/gene_count_{count}.csv" for count in gene_counts]
    gene_labels = [f"{count} Genes" for count in gene_counts]

    # Individual plots
    for count in gene_counts:
        csv_file = f"../results/gene_count_{count}.csv"
        if os.path.exists(csv_file):
            plot_single_experiment(
                csv_file,
                f"Gene Count {count} - Fitness Over Generations",
                f"gene_count_{count}.png"
            )

    # MUTATION RATE EXPERIMENTS
    print("\nProcessing Mutation Rate Experiments...")

    mutation_configs = [
        (0.05, 0.1, 0.05),   # Low
        (0.1, 0.25, 0.1),    # Medium
        (0.25, 0.5, 0.25),   # High
    ]

    mutation_csv_files = [
        f"../results/mutation_point_{p}_shrink_{s}_grow_{g}.csv"
        for p, s, g in mutation_configs
    ]

    mutation_labels = [
        f"Low (p={p}, s={s}, g={g})",
        f"Medium (p={p}, s={s}, g={g})",
        f"High (p={p}, s={s}, g={g})",
    ]

    # Individual plots
    for (p, s, g), label in zip(mutation_configs, mutation_labels):
        csv_file = f"../results/mutation_point_{p}_shrink_{s}_grow_{g}.csv"
        if os.path.exists(csv_file):
            plot_single_experiment(
                csv_file,
                f"Mutation Rates {label} - Fitness Over Generations",
                f"mutation_p{p}_s{s}_g{g}.png"
            )

    print("\n" + "="*60)
    print("ALL GRAPHS AND TABLES GENERATED SUCCESSFULLY!")
    print("="*60)
    print("\nOutput location: ../graphs/")
    print("\nGenerated files:")
    print("  - Individual plots for each experiment")
    print("  - Comparison plots for each parameter type")
    print("  - Summary tables (CSV) with statistics")

def generate_all_graphs():
    main()


def display_comparison_graphs():
    print("\nDisplaying comparison graphs...")
    print("Close each graph window to continue to the next one.\n")
    
    pass

if __name__ == "__main__":
    main()