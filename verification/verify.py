"""
Precise Graph-Based Table Comparison System with Batch Processing
Provides accumulated metrics across multiple document pairs
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from scipy.optimize import linear_sum_assignment
from difflib import SequenceMatcher
import Levenshtein
import json
import csv
from pathlib import Path
from collections import defaultdict
import statistics


@dataclass
class CellNode:
    """Represents a table cell as a graph node"""
    row: int
    col: int
    value: str
    node_id: str
    cell_type: str  # 'header', 'data', 'empty'


@dataclass
class ComparisonMetrics:
    """Comprehensive comparison metrics for a single pair"""
    # File identification
    generated_file: str
    ground_truth_file: str

    # Structural metrics
    dimension_similarity: float
    topology_similarity: float
    graph_edit_distance: float
    degree_distribution_similarity: float
    edge_similarity: float

    # Semantic metrics
    exact_match_rate: float
    fuzzy_match_rate: float
    character_level_similarity: float
    token_level_similarity: float
    position_weighted_similarity: float

    # Combined scores
    structural_score: float
    semantic_score: float
    overall_score: float

    # Counts
    total_cells_generated: int
    total_cells_ground_truth: int
    matched_cells: int
    exact_matches: int
    fuzzy_matches: int
    mismatches: int

    # Detailed breakdowns
    cell_comparisons: List[Dict]
    mismatched_cells: List[Dict]


@dataclass
class AggregatedMetrics:
    """Accumulated metrics across all document pairs"""
    # Summary
    total_pairs: int
    successful_comparisons: int
    failed_comparisons: int

    # Structural statistics
    mean_dimension_similarity: float
    std_dimension_similarity: float
    median_dimension_similarity: float
    min_dimension_similarity: float
    max_dimension_similarity: float

    mean_topology_similarity: float
    std_topology_similarity: float

    mean_graph_edit_distance: float
    std_graph_edit_distance: float

    mean_degree_distribution_similarity: float
    std_degree_distribution_similarity: float

    mean_structural_score: float
    std_structural_score: float
    median_structural_score: float

    # Semantic statistics
    mean_exact_match_rate: float
    std_exact_match_rate: float
    median_exact_match_rate: float
    min_exact_match_rate: float
    max_exact_match_rate: float

    mean_fuzzy_match_rate: float
    std_fuzzy_match_rate: float

    mean_character_similarity: float
    std_character_similarity: float

    mean_token_similarity: float
    std_token_similarity: float

    mean_position_weighted_similarity: float
    std_position_weighted_similarity: float

    mean_semantic_score: float
    std_semantic_score: float
    median_semantic_score: float

    # Overall statistics
    mean_overall_score: float
    std_overall_score: float
    median_overall_score: float
    min_overall_score: float
    max_overall_score: float

    # Aggregate counts
    total_cells_processed: int
    total_exact_matches: int
    total_fuzzy_matches: int
    total_mismatches: int

    # Quality distribution
    excellent_count: int  # >= 95%
    very_good_count: int  # >= 90%
    good_count: int  # >= 80%
    acceptable_count: int  # >= 70%
    poor_count: int  # < 70%

    # Per-pair results
    individual_results: List[ComparisonMetrics]

    # Worst performers (for investigation)
    worst_overall: List[Tuple[str, str, float]]  # (gen_file, gt_file, score)
    worst_structural: List[Tuple[str, str, float]]
    worst_semantic: List[Tuple[str, str, float]]


class TableGraphComparator:
    """Main comparator class with precise scoring and batch processing"""

    def __init__(self, config: Dict = None):
        self.config = config or {
            'structural_weight': 0.40,
            'semantic_weight': 0.60,
            'fuzzy_threshold': 0.85,
            'position_weight_decay': 0.95,
            'normalization': 'minmax'
        }

    def load_table(self, filepath: str) -> List[List[str]]:
        """Load TSV file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            table = [row for row in reader]

        return table

    def build_graph(self, table: List[List[str]]) -> nx.DiGraph:
        """Convert table to directed graph with rich node attributes"""
        G = nx.DiGraph()
        rows, cols = len(table), len(table[0]) if table else 0

        # Add nodes with attributes
        for i, row in enumerate(table):
            for j, cell in enumerate(row):
                node_id = f"r{i}c{j}"
                cell_type = 'header' if i == 0 else ('empty' if not cell.strip() else 'data')

                G.add_node(node_id,
                           row=i,
                           col=j,
                           value=cell.strip(),
                           cell_type=cell_type,
                           degree_centrality=0)

        # Add edges (8-connectivity)
        directions = [
            (0, 1, 'right'), (0, -1, 'left'),
            (1, 0, 'down'), (-1, 0, 'up'),
            (1, 1, 'diag_dr'), (-1, -1, 'diag_ul'),
            (1, -1, 'diag_dl'), (-1, 1, 'diag_ur')
        ]

        for i in range(rows):
            for j in range(cols if i < len(table) else 0):
                node_id = f"r{i}c{j}"
                for di, dj, direction in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        neighbor_id = f"r{ni}c{nj}"
                        G.add_edge(node_id, neighbor_id, direction=direction)

        # Calculate centrality
        centrality = nx.degree_centrality(G)
        nx.set_node_attributes(G, centrality, 'degree_centrality')

        return G

    def compute_dimension_similarity(self, g1: nx.DiGraph, g2: nx.DiGraph) -> Dict:
        """Precise dimension comparison"""
        nodes1 = [d for n, d in g1.nodes(data=True)]
        nodes2 = [d for n, d in g2.nodes(data=True)]

        rows1 = max([n['row'] for n in nodes1]) + 1 if nodes1 else 0
        cols1 = max([n['col'] for n in nodes1]) + 1 if nodes1 else 0
        rows2 = max([n['row'] for n in nodes2]) + 1 if nodes2 else 0
        cols2 = max([n['col'] for n in nodes2]) + 1 if nodes2 else 0

        row_similarity = 1 - abs(rows1 - rows2) / max(rows1, rows2) if max(rows1, rows2) > 0 else 1.0
        col_similarity = 1 - abs(cols1 - cols2) / max(cols1, cols2) if max(cols1, cols2) > 0 else 1.0

        dimension_similarity = (row_similarity + col_similarity) / 2

        return {
            'similarity': dimension_similarity,
            'rows': (rows1, rows2),
            'cols': (cols1, cols2),
            'row_diff': abs(rows1 - rows2),
            'col_diff': abs(cols1 - cols2)
        }

    def compute_topology_similarity(self, g1: nx.DiGraph, g2: nx.DiGraph) -> Dict:
        """Compare graph topology using multiple metrics"""

        # Edge count similarity
        edge_sim = 1 - abs(g1.number_of_edges() - g2.number_of_edges()) / max(g1.number_of_edges(),
                                                                              g2.number_of_edges(), 1)

        # Node count similarity
        node_sim = 1 - abs(g1.number_of_nodes() - g2.number_of_nodes()) / max(g1.number_of_nodes(),
                                                                              g2.number_of_nodes(), 1)

        # Degree distribution similarity (Jensen-Shannon divergence)
        deg1 = sorted([d for n, d in g1.degree()])
        deg2 = sorted([d for n, d in g2.degree()])
        max_deg = max(max(deg1, default=0), max(deg2, default=0))

        if max_deg > 0:
            hist1 = np.histogram(deg1, bins=range(max_deg + 2))[0]
            hist2 = np.histogram(deg2, bins=range(max_deg + 2))[0]

            # Normalize
            hist1 = hist1 / hist1.sum() if hist1.sum() > 0 else hist1
            hist2 = hist2 / hist2.sum() if hist2.sum() > 0 else hist2

            # Jensen-Shannon Divergence
            m = (hist1 + hist2) / 2
            jsd = 0
            for p, q, mi in zip(hist1, hist2, m):
                if p > 0:
                    jsd += p * np.log2(p / mi) if mi > 0 else 0
                if q > 0:
                    jsd += q * np.log2(q / mi) if mi > 0 else 0
            jsd = jsd / 2

            degree_sim = 1 - np.sqrt(jsd)
        else:
            degree_sim = 1.0

        topology_similarity = (edge_sim * 0.3 + node_sim * 0.3 + degree_sim * 0.4)

        return {
            'similarity': topology_similarity,
            'edge_similarity': edge_sim,
            'node_similarity': node_sim,
            'degree_distribution_similarity': degree_sim,
            'edges': (g1.number_of_edges(), g2.number_of_edges()),
            'nodes': (g1.number_of_nodes(), g2.number_of_nodes())
        }

    def compute_graph_edit_distance(self, g1: nx.DiGraph, g2: nx.DiGraph) -> Dict:
        """Compute normalized Graph Edit Distance"""
        max_nodes = max(g1.number_of_nodes(), g2.number_of_nodes())

        if max_nodes < 100:
            try:
                ged = nx.graph_edit_distance(g1, g2, timeout=30)
            except:
                ged = self._approximate_ged(g1, g2)
        else:
            ged = self._approximate_ged(g1, g2)

        max_ged = max_nodes + max(g1.number_of_edges(), g2.number_of_edges())
        normalized_ged = 1 - (ged / max_ged) if max_ged > 0 else 1.0

        return {
            'similarity': normalized_ged,
            'raw_ged': ged,
            'normalized_ged': normalized_ged
        }

    def _approximate_ged(self, g1: nx.DiGraph, g2: nx.DiGraph) -> float:
        """Approximate GED using node/edge differences"""
        node_diff = abs(g1.number_of_nodes() - g2.number_of_nodes())
        edge_diff = abs(g1.number_of_edges() - g2.number_of_edges())
        return node_diff + edge_diff

    def compute_semantic_similarity(self, g1: nx.DiGraph, g2: nx.DiGraph) -> Dict:
        """Optimal cell matching with multiple semantic metrics"""
        nodes1 = [(n, d) for n, d in g1.nodes(data=True)]
        nodes2 = [(n, d) for n, d in g2.nodes(data=True)]

        # Build cost matrix for Hungarian algorithm
        n1, n2 = len(nodes1), len(nodes2)
        max_dim = max(n1, n2)
        cost_matrix = np.ones((max_dim, max_dim)) * 1000

        for i, (id1, data1) in enumerate(nodes1):
            for j, (id2, data2) in enumerate(nodes2):
                pos_penalty = abs(data1['row'] - data2['row']) + abs(data1['col'] - data2['col'])
                content_sim = self._compute_cell_similarity(data1['value'], data2['value'])
                cost_matrix[i, j] = (1 - content_sim) * 10 + pos_penalty * 0.1

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Compute detailed metrics
        cell_comparisons = []
        exact_matches = 0
        fuzzy_matches = 0
        total_char_similarity = 0
        total_token_similarity = 0
        position_weighted_sum = 0
        total_weight = 0

        for i, j in zip(row_ind, col_ind):
            if i < n1 and j < n2:
                id1, data1 = nodes1[i]
                id2, data2 = nodes2[j]

                val1, val2 = data1['value'], data2['value']

                char_sim = self._levenshtein_similarity(val1, val2)
                token_sim = self._token_similarity(val1, val2)
                exact = 1.0 if val1.lower() == val2.lower() else 0.0
                fuzzy = 1.0 if char_sim >= self.config['fuzzy_threshold'] else 0.0

                pos_weight = self.config['position_weight_decay'] ** data1['row']

                exact_matches += exact
                fuzzy_matches += fuzzy
                total_char_similarity += char_sim
                total_token_similarity += token_sim
                position_weighted_sum += char_sim * pos_weight
                total_weight += pos_weight

                if char_sim < 1.0:
                    cell_comparisons.append({
                        'position': (data1['row'], data1['col']),
                        'generated': val1,
                        'ground_truth': val2,
                        'char_similarity': round(char_sim, 4),
                        'token_similarity': round(token_sim, 4),
                        'exact_match': bool(exact),
                        'fuzzy_match': bool(fuzzy)
                    })

        matched_cells = min(n1, n2)

        return {
            'exact_match_rate': exact_matches / matched_cells if matched_cells > 0 else 0,
            'fuzzy_match_rate': fuzzy_matches / matched_cells if matched_cells > 0 else 0,
            'character_similarity': total_char_similarity / matched_cells if matched_cells > 0 else 0,
            'token_similarity': total_token_similarity / matched_cells if matched_cells > 0 else 0,
            'position_weighted_similarity': position_weighted_sum / total_weight if total_weight > 0 else 0,
            'matched_cells': matched_cells,
            'total_cells_g1': n1,
            'total_cells_g2': n2,
            'exact_matches': int(exact_matches),
            'fuzzy_matches': int(fuzzy_matches),
            'mismatches': int(matched_cells - fuzzy_matches),
            'mismatched_cells': cell_comparisons
        }

    def _compute_cell_similarity(self, val1: str, val2: str) -> float:
        """Combined cell similarity metric"""
        if not val1 and not val2:
            return 1.0
        if not val1 or not val2:
            return 0.0

        char_sim = self._levenshtein_similarity(val1, val2)
        token_sim = self._token_similarity(val1, val2)

        return (char_sim * 0.6 + token_sim * 0.4)

    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Normalized Levenshtein similarity"""
        if not s1 and not s2:
            return 1.0
        distance = Levenshtein.distance(s1.lower(), s2.lower())
        max_len = max(len(s1), len(s2))
        return 1 - (distance / max_len) if max_len > 0 else 1.0

    def _token_similarity(self, s1: str, s2: str) -> float:
        """Token-based similarity (word level)"""
        tokens1 = set(s1.lower().split())
        tokens2 = set(s2.lower().split())

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def compare_single(self, generated_path: str, ground_truth_path: str) -> ComparisonMetrics:
        """Compare a single pair of tables"""
        # Load tables
        table1 = self.load_table(generated_path)
        table2 = self.load_table(ground_truth_path)

        # Build graphs
        g1 = self.build_graph(table1)
        g2 = self.build_graph(table2)

        # Structural analysis
        dim_metrics = self.compute_dimension_similarity(g1, g2)
        topo_metrics = self.compute_topology_similarity(g1, g2)
        ged_metrics = self.compute_graph_edit_distance(g1, g2)

        # Semantic analysis
        sem_metrics = self.compute_semantic_similarity(g1, g2)

        # Combine scores
        structural_score = (
                dim_metrics['similarity'] * 0.30 +
                topo_metrics['similarity'] * 0.30 +
                ged_metrics['similarity'] * 0.25 +
                topo_metrics['degree_distribution_similarity'] * 0.15
        )

        semantic_score = (
                sem_metrics['position_weighted_similarity'] * 0.40 +
                sem_metrics['character_similarity'] * 0.30 +
                sem_metrics['token_similarity'] * 0.20 +
                sem_metrics['fuzzy_match_rate'] * 0.10
        )

        overall_score = (
                structural_score * self.config['structural_weight'] +
                semantic_score * self.config['semantic_weight']
        )

        return ComparisonMetrics(
            generated_file=Path(generated_path).name,
            ground_truth_file=Path(ground_truth_path).name,
            dimension_similarity=dim_metrics['similarity'],
            topology_similarity=topo_metrics['similarity'],
            graph_edit_distance=ged_metrics['similarity'],
            degree_distribution_similarity=topo_metrics['degree_distribution_similarity'],
            edge_similarity=topo_metrics['edge_similarity'],
            exact_match_rate=sem_metrics['exact_match_rate'],
            fuzzy_match_rate=sem_metrics['fuzzy_match_rate'],
            character_level_similarity=sem_metrics['character_similarity'],
            token_level_similarity=sem_metrics['token_similarity'],
            position_weighted_similarity=sem_metrics['position_weighted_similarity'],
            structural_score=structural_score,
            semantic_score=semantic_score,
            overall_score=overall_score,
            total_cells_generated=sem_metrics['total_cells_g1'],
            total_cells_ground_truth=sem_metrics['total_cells_g2'],
            matched_cells=sem_metrics['matched_cells'],
            exact_matches=sem_metrics['exact_matches'],
            fuzzy_matches=sem_metrics['fuzzy_matches'],
            mismatches=sem_metrics['mismatches'],
            cell_comparisons=sem_metrics['mismatched_cells'],
            mismatched_cells=[c for c in sem_metrics['mismatched_cells'] if not c['fuzzy_match']]
        )

    def compare_batch(self, pairs: List[Tuple[str, str]], verbose: bool = True) -> AggregatedMetrics:
        """
        Compare multiple document pairs and return aggregated metrics

        Args:
            pairs: List of (generated_path, ground_truth_path) tuples
            verbose: Print progress

        Returns:
            AggregatedMetrics with accumulated statistics
        """
        individual_results = []
        failed_pairs = []

        for idx, (gen_path, gt_path) in enumerate(pairs):
            if verbose:
                print(f"Processing pair {idx + 1}/{len(pairs)}: {Path(gen_path).name}")

            try:
                metrics = self.compare_single(gen_path, gt_path)
                individual_results.append(metrics)
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                failed_pairs.append((gen_path, gt_path, str(e)))

        # Aggregate statistics
        if not individual_results:
            raise ValueError("No successful comparisons to aggregate")

        return self._aggregate_metrics(individual_results, len(failed_pairs))

    def _aggregate_metrics(self, results: List[ComparisonMetrics], failed_count: int) -> AggregatedMetrics:
        """Compute aggregated statistics from individual results"""

        # Extract all metrics
        dim_sims = [r.dimension_similarity for r in results]
        topo_sims = [r.topology_similarity for r in results]
        ged_sims = [r.graph_edit_distance for r in results]
        deg_sims = [r.degree_distribution_similarity for r in results]
        struct_scores = [r.structural_score for r in results]

        exact_rates = [r.exact_match_rate for r in results]
        fuzzy_rates = [r.fuzzy_match_rate for r in results]
        char_sims = [r.character_level_similarity for r in results]
        token_sims = [r.token_level_similarity for r in results]
        pos_sims = [r.position_weighted_similarity for r in results]
        sem_scores = [r.semantic_score for r in results]

        overall_scores = [r.overall_score for r in results]

        # Quality distribution
        excellent = sum(1 for s in overall_scores if s >= 0.95)
        very_good = sum(1 for s in overall_scores if 0.90 <= s < 0.95)
        good = sum(1 for s in overall_scores if 0.80 <= s < 0.90)
        acceptable = sum(1 for s in overall_scores if 0.70 <= s < 0.80)
        poor = sum(1 for s in overall_scores if s < 0.70)

        # Aggregate counts
        total_cells = sum(r.matched_cells for r in results)
        total_exact = sum(r.exact_matches for r in results)
        total_fuzzy = sum(r.fuzzy_matches for r in results)
        total_mismatch = sum(r.mismatches for r in results)

        # Worst performers (for debugging)
        sorted_by_overall = sorted(results, key=lambda x: x.overall_score)
        sorted_by_struct = sorted(results, key=lambda x: x.structural_score)
        sorted_by_sem = sorted(results, key=lambda x: x.semantic_score)

        worst_overall = [(r.generated_file, r.ground_truth_file, r.overall_score)
                         for r in sorted_by_overall[:5]]
        worst_structural = [(r.generated_file, r.ground_truth_file, r.structural_score)
                            for r in sorted_by_struct[:5]]
        worst_semantic = [(r.generated_file, r.ground_truth_file, r.semantic_score)
                          for r in sorted_by_sem[:5]]

        return AggregatedMetrics(
            total_pairs=len(results) + failed_count,
            successful_comparisons=len(results),
            failed_comparisons=failed_count,

            # Structural stats
            mean_dimension_similarity=statistics.mean(dim_sims),
            std_dimension_similarity=statistics.stdev(dim_sims) if len(dim_sims) > 1 else 0,
            median_dimension_similarity=statistics.median(dim_sims),
            min_dimension_similarity=min(dim_sims),
            max_dimension_similarity=max(dim_sims),

            mean_topology_similarity=statistics.mean(topo_sims),
            std_topology_similarity=statistics.stdev(topo_sims) if len(topo_sims) > 1 else 0,

            mean_graph_edit_distance=statistics.mean(ged_sims),
            std_graph_edit_distance=statistics.stdev(ged_sims) if len(ged_sims) > 1 else 0,

            mean_degree_distribution_similarity=statistics.mean(deg_sims),
            std_degree_distribution_similarity=statistics.stdev(deg_sims) if len(deg_sims) > 1 else 0,

            mean_structural_score=statistics.mean(struct_scores),
            std_structural_score=statistics.stdev(struct_scores) if len(struct_scores) > 1 else 0,
            median_structural_score=statistics.median(struct_scores),

            # Semantic stats
            mean_exact_match_rate=statistics.mean(exact_rates),
            std_exact_match_rate=statistics.stdev(exact_rates) if len(exact_rates) > 1 else 0,
            median_exact_match_rate=statistics.median(exact_rates),
            min_exact_match_rate=min(exact_rates),
            max_exact_match_rate=max(exact_rates),

            mean_fuzzy_match_rate=statistics.mean(fuzzy_rates),
            std_fuzzy_match_rate=statistics.stdev(fuzzy_rates) if len(fuzzy_rates) > 1 else 0,

            mean_character_similarity=statistics.mean(char_sims),
            std_character_similarity=statistics.stdev(char_sims) if len(char_sims) > 1 else 0,

            mean_token_similarity=statistics.mean(token_sims),
            std_token_similarity=statistics.stdev(token_sims) if len(token_sims) > 1 else 0,

            mean_position_weighted_similarity=statistics.mean(pos_sims),
            std_position_weighted_similarity=statistics.stdev(pos_sims) if len(pos_sims) > 1 else 0,

            mean_semantic_score=statistics.mean(sem_scores),
            std_semantic_score=statistics.stdev(sem_scores) if len(sem_scores) > 1 else 0,
            median_semantic_score=statistics.median(sem_scores),

            # Overall stats
            mean_overall_score=statistics.mean(overall_scores),
            std_overall_score=statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
            median_overall_score=statistics.median(overall_scores),
            min_overall_score=min(overall_scores),
            max_overall_score=max(overall_scores),

            # Counts
            total_cells_processed=total_cells,
            total_exact_matches=total_exact,
            total_fuzzy_matches=total_fuzzy,
            total_mismatches=total_mismatch,

            # Quality distribution
            excellent_count=excellent,
            very_good_count=very_good,
            good_count=good,
            acceptable_count=acceptable,
            poor_count=poor,

            # Detailed results
            individual_results=results,
            worst_overall=worst_overall,
            worst_structural=worst_structural,
            worst_semantic=worst_semantic
        )

    def generate_batch_report(self, agg_metrics: AggregatedMetrics, output_path: str = 'batch_comparison_report.json'):
        """Generate comprehensive batch report"""

        report = {
            'summary': {
                'total_pairs': agg_metrics.total_pairs,
                'successful': agg_metrics.successful_comparisons,
                'failed': agg_metrics.failed_comparisons,
                'success_rate': agg_metrics.successful_comparisons / agg_metrics.total_pairs * 100
            },

            'overall_performance': {
                'mean_score': round(agg_metrics.mean_overall_score * 100, 2),
                'std_score': round(agg_metrics.std_overall_score * 100, 2),
                'median_score': round(agg_metrics.median_overall_score * 100, 2),
                'min_score': round(agg_metrics.min_overall_score * 100, 2),
                'max_score': round(agg_metrics.max_overall_score * 100, 2),
                'quality_assessment': self._assess_quality(agg_metrics.mean_overall_score)
            },

            'structural_analysis': {
                'mean_structural_score': round(agg_metrics.mean_structural_score * 100, 2),
                'std_structural_score': round(agg_metrics.std_structural_score * 100, 2),
                'median_structural_score': round(agg_metrics.median_structural_score * 100, 2),
                'dimension_similarity': {
                    'mean': round(agg_metrics.mean_dimension_similarity * 100, 2),
                    'std': round(agg_metrics.std_dimension_similarity * 100, 2),
                    'median': round(agg_metrics.median_dimension_similarity * 100, 2),
                    'min': round(agg_metrics.min_dimension_similarity * 100, 2),
                    'max': round(agg_metrics.max_dimension_similarity * 100, 2)
                },
                'topology_similarity': {
                    'mean': round(agg_metrics.mean_topology_similarity * 100, 2),
                    'std': round(agg_metrics.std_topology_similarity * 100, 2)
                },
                'graph_edit_distance': {
                    'mean': round(agg_metrics.mean_graph_edit_distance * 100, 2),
                    'std': round(agg_metrics.std_graph_edit_distance * 100, 2)
                },
                'degree_distribution_similarity': {
                    'mean': round(agg_metrics.mean_degree_distribution_similarity * 100, 2),
                    'std': round(agg_metrics.std_degree_distribution_similarity * 100, 2)
                }
            },

            'semantic_analysis': {
                'mean_semantic_score': round(agg_metrics.mean_semantic_score * 100, 2),
                'std_semantic_score': round(agg_metrics.std_semantic_score * 100, 2),
                'median_semantic_score': round(agg_metrics.median_semantic_score * 100, 2),
                'exact_match_rate': {
                    'mean': round(agg_metrics.mean_exact_match_rate * 100, 2),
                    'std': round(agg_metrics.std_exact_match_rate * 100, 2),
                    'median': round(agg_metrics.median_exact_match_rate * 100, 2),
                    'min': round(agg_metrics.min_exact_match_rate * 100, 2),
                    'max': round(agg_metrics.max_exact_match_rate * 100, 2)
                },
                'fuzzy_match_rate': {
                    'mean': round(agg_metrics.mean_fuzzy_match_rate * 100, 2),
                    'std': round(agg_metrics.std_fuzzy_match_rate * 100, 2)
                },
                'character_similarity': {
                    'mean': round(agg_metrics.mean_character_similarity * 100, 2),
                    'std': round(agg_metrics.std_character_similarity * 100, 2)
                },
                'token_similarity': {
                    'mean': round(agg_metrics.mean_token_similarity * 100, 2),
                    'std': round(agg_metrics.std_token_similarity * 100, 2)
                },
                'position_weighted_similarity': {
                    'mean': round(agg_metrics.mean_position_weighted_similarity * 100, 2),
                    'std': round(agg_metrics.std_position_weighted_similarity * 100, 2)
                }
            },

            'cell_statistics': {
                'total_cells_processed': agg_metrics.total_cells_processed,
                'total_exact_matches': agg_metrics.total_exact_matches,
                'total_fuzzy_matches': agg_metrics.total_fuzzy_matches,
                'total_mismatches': agg_metrics.total_mismatches,
                'overall_exact_match_rate': round(
                    agg_metrics.total_exact_matches / agg_metrics.total_cells_processed * 100,
                    2) if agg_metrics.total_cells_processed > 0 else 0
            },

            'quality_distribution': {
                'excellent_95plus': agg_metrics.excellent_count,
                'very_good_90to95': agg_metrics.very_good_count,
                'good_80to90': agg_metrics.good_count,
                'acceptable_70to80': agg_metrics.acceptable_count,
                'poor_below70': agg_metrics.poor_count,
                'distribution_percentages': {
                    'excellent': round(agg_metrics.excellent_count / agg_metrics.successful_comparisons * 100, 2),
                    'very_good': round(agg_metrics.very_good_count / agg_metrics.successful_comparisons * 100, 2),
                    'good': round(agg_metrics.good_count / agg_metrics.successful_comparisons * 100, 2),
                    'acceptable': round(agg_metrics.acceptable_count / agg_metrics.successful_comparisons * 100, 2),
                    'poor': round(agg_metrics.poor_count / agg_metrics.successful_comparisons * 100, 2)
                }
            },

            'worst_performers': {
                'by_overall_score': [
                    {'generated': gen, 'ground_truth': gt, 'score': round(score * 100, 2)}
                    for gen, gt, score in agg_metrics.worst_overall
                ],
                'by_structural_score': [
                    {'generated': gen, 'ground_truth': gt, 'score': round(score * 100, 2)}
                    for gen, gt, score in agg_metrics.worst_structural
                ],
                'by_semantic_score': [
                    {'generated': gen, 'ground_truth': gt, 'score': round(score * 100, 2)}
                    for gen, gt, score in agg_metrics.worst_semantic
                ]
            },

            'individual_results': [
                {
                    'generated_file': r.generated_file,
                    'ground_truth_file': r.ground_truth_file,
                    'overall_score': round(r.overall_score * 100, 2),
                    'structural_score': round(r.structural_score * 100, 2),
                    'semantic_score': round(r.semantic_score * 100, 2),
                    'exact_match_rate': round(r.exact_match_rate * 100, 2),
                    'mismatches': r.mismatches
                }
                for r in agg_metrics.individual_results
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def _assess_quality(self, score: float) -> str:
        """Quality assessment based on score"""
        if score >= 0.95:
            return "Excellent - Production Ready"
        elif score >= 0.90:
            return "Very Good - Minor review needed"
        elif score >= 0.80:
            return "Good - Some corrections required"
        elif score >= 0.70:
            return "Acceptable - Significant review needed"
        else:
            return "Poor - Major corrections required"

    def print_batch_summary(self, agg_metrics: AggregatedMetrics):
        """Print human-readable summary of batch comparison"""
        print("=" * 80)
        print("BATCH COMPARISON SUMMARY")
        print("=" * 80)
        print(f"\nTotal Pairs: {agg_metrics.total_pairs}")
        print(f"Successful: {agg_metrics.successful_comparisons}")
        print(f"Failed: {agg_metrics.failed_comparisons}")
        print(f"Success Rate: {agg_metrics.successful_comparisons / agg_metrics.total_pairs * 100:.2f}%")

        print(f"\n{'OVERALL PERFORMANCE':-^80}")
        print(
            f"Mean Score:   {agg_metrics.mean_overall_score * 100:6.2f}% ± {agg_metrics.std_overall_score * 100:.2f}%")
        print(f"Median Score: {agg_metrics.median_overall_score * 100:6.2f}%")
        print(f"Range:        {agg_metrics.min_overall_score * 100:6.2f}% - {agg_metrics.max_overall_score * 100:.2f}%")
        print(f"Assessment:   {self._assess_quality(agg_metrics.mean_overall_score)}")

        print(f"\n{'STRUCTURAL ANALYSIS':-^80}")
        print(
            f"Mean Structural Score: {agg_metrics.mean_structural_score * 100:6.2f}% ± {agg_metrics.std_structural_score * 100:.2f}%")
        print(f"  - Dimension Similarity:  {agg_metrics.mean_dimension_similarity * 100:6.2f}%")
        print(f"  - Topology Similarity:   {agg_metrics.mean_topology_similarity * 100:6.2f}%")
        print(f"  - Graph Edit Distance:   {agg_metrics.mean_graph_edit_distance * 100:6.2f}%")

        print(f"\n{'SEMANTIC ANALYSIS':-^80}")
        print(
            f"Mean Semantic Score: {agg_metrics.mean_semantic_score * 100:6.2f}% ± {agg_metrics.std_semantic_score * 100:.2f}%")
        print(f"  - Exact Match Rate:      {agg_metrics.mean_exact_match_rate * 100:6.2f}%")
        print(f"  - Fuzzy Match Rate:      {agg_metrics.mean_fuzzy_match_rate * 100:6.2f}%")
        print(f"  - Character Similarity:  {agg_metrics.mean_character_similarity * 100:6.2f}%")

        print(f"\n{'CELL STATISTICS':-^80}")
        print(f"Total Cells Processed: {agg_metrics.total_cells_processed:,}")
        print(
            f"Exact Matches:         {agg_metrics.total_exact_matches:,} ({agg_metrics.total_exact_matches / agg_metrics.total_cells_processed * 100:.2f}%)")
        print(
            f"Fuzzy Matches:         {agg_metrics.total_fuzzy_matches:,} ({agg_metrics.total_fuzzy_matches / agg_metrics.total_cells_processed * 100:.2f}%)")
        print(
            f"Mismatches:            {agg_metrics.total_mismatches:,} ({agg_metrics.total_mismatches / agg_metrics.total_cells_processed * 100:.2f}%)")

        print(f"\n{'QUALITY DISTRIBUTION':-^80}")
        total = agg_metrics.successful_comparisons
        print(
            f"Excellent (≥95%):    {agg_metrics.excellent_count:3d} ({agg_metrics.excellent_count / total * 100:5.1f}%)")
        print(
            f"Very Good (≥90%):    {agg_metrics.very_good_count:3d} ({agg_metrics.very_good_count / total * 100:5.1f}%)")
        print(f"Good (≥80%):         {agg_metrics.good_count:3d} ({agg_metrics.good_count / total * 100:5.1f}%)")
        print(
            f"Acceptable (≥70%):   {agg_metrics.acceptable_count:3d} ({agg_metrics.acceptable_count / total * 100:5.1f}%)")
        print(f"Poor (<70%):         {agg_metrics.poor_count:3d} ({agg_metrics.poor_count / total * 100:5.1f}%)")

        if agg_metrics.worst_overall:
            print(f"\n{'WORST PERFORMERS (Overall Score)':-^80}")
            for gen, gt, score in agg_metrics.worst_overall:
                print(f"  {score * 100:5.2f}% - {gen} vs {gt}")

        print("=" * 80)

