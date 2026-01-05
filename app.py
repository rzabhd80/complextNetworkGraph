import os

from graph.visualize import TableGraphBuilder, visualize
from pipeline import run_pipeline
from verification.verify import TableGraphComparator

if __name__ == "__main__":
   pairs = [
      ('./reconstruct/natwest_1.tsv', './verification/ground_truth/natwest_1.tsv'),
      ('./reconstruct/natwest_2.tsv', './verification/ground_truth/natwest_2.tsv'),
      ('./reconstruct/natwest_3.tsv', './verification/ground_truth/natwest_3.tsv'),
      ('./reconstruct/BankTSB_1.tsv', './verification/ground_truth/BankTSB_1.tsv'),
      ('./reconstruct/BankTSB_2.tsv', './verification/ground_truth/BankTSB_2.tsv'),
   ]

   comparator = TableGraphComparator(config={
      'structural_weight': 0.40,
      'semantic_weight': 0.60,
      'fuzzy_threshold': 0.85,
      'position_weight_decay': 0.95
   })
   # Run batch comparison
   agg_metrics = comparator.compare_batch(pairs, verbose=True)

   # Print summary
   comparator.print_batch_summary(agg_metrics)

   # Generate detailed report
   report = comparator.generate_batch_report(agg_metrics, 'batch_report.json')
   print(f"\nDetailed report saved to: batch_report.json")

   # Access specific metrics
   print(f"\n\nKey Metrics:")
   print(f"  Mean Overall Score: {agg_metrics.mean_overall_score * 100:.2f}%")
   print(f"  Standard Deviation: {agg_metrics.std_overall_score * 100:.2f}%")
   print(f"  Best Score: {agg_metrics.max_overall_score * 100:.2f}%")
   print(f"  Worst Score: {agg_metrics.min_overall_score * 100:.2f}%")
   print(f"  Total Cells: {agg_metrics.total_cells_processed:,}")
   print(f"  Overall Accuracy: {agg_metrics.total_exact_matches / agg_metrics.total_cells_processed * 100:.2f}%")
