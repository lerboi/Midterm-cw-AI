import sys
import os

# Add experiments folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

def main():    
    print("="*60)
    print(" "*15 + "MOUNTAIN CLIMBING GA - FULL EXPERIMENT")
    print("="*60)
    print("Experiments to run:")
    print("  1. Population Size (5, 10, 20, 50) - 4 runs x 500 generations")
    print("  2. Gene Count (2, 3, 5, 10) - 4 runs x 500 generations")
    print("  3. Mutation Rates (Low, Medium, High) - 3 runs x 500 generations")
    print()
    
    input("Press ENTER to start experiments, or Ctrl+C to cancel...")
    print()
    
    # ========================================
    # STEP 1: Population Size Experiments
    # ========================================
    print("="*70)
    print("STEP 1/4: Running Population Size Experiments")
    print("="*70)
    print("Testing population sizes: 5, 10, 20, 50")
    print()
    
    try:
        from experiment_population_size import run_population_experiments
        run_population_experiments()
        print("\n✓ Population size experiments completed successfully!\n")
    except Exception as e:
        print(f"\n✗ Error in population size experiments: {e}\n")
        print("Continuing with remaining experiments...\n")
    
    # ========================================
    # STEP 2: Gene Count Experiments
    # ========================================
    print("="*70)
    print("STEP 2/4: Running Gene Count Experiments")
    print("="*70)
    print("Testing gene counts: 2, 3, 5, 10")
    print()
    
    try:
        from experiment_gene_count import run_gene_experiments
        run_gene_experiments()
        print("\n✓ Gene count experiments completed successfully!\n")
    except Exception as e:
        print(f"\n✗ Error in gene count experiments: {e}\n")
        print("Continuing with remaining experiments...\n")
    
    # ========================================
    # STEP 3: Mutation Rate Experiments
    # ========================================
    print("="*70)
    print("STEP 3/4: Running Mutation Rate Experiments")
    print("="*70)
    print("Testing mutation rates: Low, Medium, High")
    print()
    
    try:
        from experiment_mutation_rates import run_mutation_experiments
        run_mutation_experiments()
        print("\n✓ Mutation rate experiments completed successfully!\n")
    except Exception as e:
        print(f"\n✗ Error in mutation rate experiments: {e}\n")
        print("Continuing with visualization...\n")
    
    # ========================================
    # STEP 4: Generate Graphs and Tables
    # ========================================
    print("="*70)
    print("STEP 4/4: Generating Graphs and Tables")
    print("="*70)
    print()
    
    try:
        from create_graphs import generate_all_graphs, display_comparison_graphs
        generate_all_graphs()
        print("\n✓ Graphs and tables generated successfully!\n")
        
        print("="*70)
        print("Displaying comparison graphs...")
        print("="*70)
        print("Close each graph window to continue to the next one.")
        print()
        display_comparison_graphs()
        
    except Exception as e:
        print(f"\n✗ Error generating graphs: {e}\n")
    
    # ========================================
    # COMPLETION SUMMARY
    # ========================================
    print("\n" + "="*70)
    print(" "*20 + "ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print()
    print("Results saved to:")
    print("  📁 results/           - CSV data files with fitness values")
    print("  📊 graphs/            - PNG graphs and summary tables")
    print("  🧬 elite_creatures/   - Best creature DNA from each experiment")
    print()
    print("Next steps:")
    print("  1. Review graphs in graphs/ folder")
    print("  2. Analyze summary tables (summary_*.csv)")
    print("  3. Test elite creatures with: python realtime_from_csv.py elite_creatures/[file].csv")
    print("  4. Include graphs and analysis in your report")
    print()
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiments cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)