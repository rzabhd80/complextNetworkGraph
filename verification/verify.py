"""
Precise Graph-Based Table Comparison System
Provides highly granular structural and semantic scoring
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from difflib import SequenceMatcher
import Levenshtein
import json
import csv


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
    """Comprehensive comparison metrics"""
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

    # Detailed breakdowns
    cell_comparisons: List[Dict]
    mismatched_cells: List[Dict]


class TableGraphComparator:
    """Main comparator class with precise scoring"""

    def __init__(self, config: Dict = None):
        self.config = config or {
            'structural_weight': 0.40,
            'semantic_weight': 0.60,
            'fuzzy_threshold': 0.85,
            'position_weight_decay': 0.95,  # Headers more important
            'normalization': 'minmax'
        }

    def load_table(self, filepath: str, delimiter: str = None) -> List[List[str]]:
        """Load CSV/TSV file"""
        if delimiter is None:
            # Auto-detect delimiter
            with open(filepath, 'r', encoding='utf-8') as f:
                sample = f.read(1024)
                delimiter = '\t' if '\t' in sample else ','

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            table = [row for row in reader]

        return table

    def build_graph(self, table: List[List[str]]) -> nx.DiGraph:
        """
        Convert table to directed graph with rich node attributes
        """
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

        # Add edges (8-connectivity: includes diagonals)
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

            degree_sim = 1 - np.sqrt(jsd)  # Convert divergence to similarity
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
        """
        Compute normalized Graph Edit Distance
        Using approximation for large graphs
        """
        max_nodes = max(g1.number_of_nodes(), g2.number_of_nodes())

        if max_nodes < 100:  # Exact GED for small graphs
            try:
                ged = nx.graph_edit_distance(g1, g2, timeout=30)
            except:
                # Fallback to approximation
                ged = self._approximate_ged(g1, g2)
        else:
            ged = self._approximate_ged(g1, g2)

        # Normalize: max possible GED is sum of nodes and edges
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
        """
        Optimal cell matching with multiple semantic metrics
        """
        nodes1 = [(n, d) for n, d in g1.nodes(data=True)]
        nodes2 = [(n, d) for n, d in g2.nodes(data=True)]

        # Build cost matrix for Hungarian algorithm
        n1, n2 = len(nodes1), len(nodes2)
        max_dim = max(n1, n2)
        cost_matrix = np.ones((max_dim, max_dim)) * 1000  # High cost for unmatched

        for i, (id1, data1) in enumerate(nodes1):
            for j, (id2, data2) in enumerate(nodes2):
                # Position penalty (prefer matching same positions)
                pos_penalty = abs(data1['row'] - data2['row']) + abs(data1['col'] - data2['col'])

                # Content similarity (inverted for cost)
                content_sim = self._compute_cell_similarity(data1['value'], data2['value'])

                # Combined cost (lower is better)
                cost_matrix[i, j] = (1 - content_sim) * 10 + pos_penalty * 0.1

        # Hungarian algorithm for optimal matching
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

                # Multiple similarity metrics
                char_sim = self._levenshtein_similarity(val1, val2)
                token_sim = self._token_similarity(val1, val2)
                exact = 1.0 if val1.lower() == val2.lower() else 0.0
                fuzzy = 1.0 if char_sim >= self.config['fuzzy_threshold'] else 0.0

                # Position weight (headers more important)
                pos_weight = self.config['position_weight_decay'] ** data1['row']

                exact_matches += exact
                fuzzy_matches += fuzzy
                total_char_similarity += char_sim
                total_token_similarity += token_sim
                position_weighted_sum += char_sim * pos_weight
                total_weight += pos_weight

                if char_sim < 1.0:  # Only log mismatches
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

    def compare(self, generated_path: str, ground_truth_path: str) -> ComparisonMetrics:
        """
        Main comparison function - returns comprehensive metrics
        """
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

        # Combine structural scores with precise weights
        structural_score = (
                dim_metrics['similarity'] * 0.30 +
                topo_metrics['similarity'] * 0.30 +
                ged_metrics['similarity'] * 0.25 +
                topo_metrics['degree_distribution_similarity'] * 0.15
        )

        # Semantic score (position-weighted is most important)
        semantic_score = (
                sem_metrics['position_weighted_similarity'] * 0.40 +
                sem_metrics['character_similarity'] * 0.30 +
                sem_metrics['token_similarity'] * 0.20 +
                sem_metrics['fuzzy_match_rate'] * 0.10
        )

        # Overall score
        overall_score = (
                structural_score * self.config['structural_weight'] +
                semantic_score * self.config['semantic_weight']
        )

        return ComparisonMetrics(
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
            cell_comparisons=sem_metrics['mismatched_cells'],
            mismatched_cells=[c for c in sem_metrics['mismatched_cells'] if not c['fuzzy_match']]
        )

    def generate_report(self, metrics: ComparisonMetrics, output_path: str = 'comparison_report.json'):
        """Generate detailed JSON report"""
        report = {
            'overall_score': round(metrics.overall_score * 100, 2),
            'structural_analysis': {
                'structural_score': round(metrics.structural_score * 100, 2),
                'dimension_similarity': round(metrics.dimension_similarity * 100, 2),
                'topology_similarity': round(metrics.topology_similarity * 100, 2),
                'graph_edit_distance_similarity': round(metrics.graph_edit_distance * 100, 2),
                'degree_distribution_similarity': round(metrics.degree_distribution_similarity * 100, 2),
                'edge_similarity': round(metrics.edge_similarity * 100, 2)
            },
            'semantic_analysis': {
                'semantic_score': round(metrics.semantic_score * 100, 2),
                'exact_match_rate': round(metrics.exact_match_rate * 100, 2),
                'fuzzy_match_rate': round(metrics.fuzzy_match_rate * 100, 2),
                'character_similarity': round(metrics.character_level_similarity * 100, 2),
                'token_similarity': round(metrics.token_level_similarity * 100, 2),
                'position_weighted_similarity': round(metrics.position_weighted_similarity * 100, 2)
            },
            'quality_assessment': self._assess_quality(metrics.overall_score),
            'mismatched_cells_count': len(metrics.mismatched_cells),
            'detailed_mismatches': metrics.mismatched_cells[:50]  # Top 50 for readability
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


# Example usage
if __name__ == "__main__":
    # Initialize comparator with custom config
    config = {
        'structural_weight': 0.40,
        'semantic_weight': 0.60,
        'fuzzy_threshold': 0.85,
        'position_weight_decay': 0.95
    }

    comparator = TableGraphComparator(config)

    # Compare tables
    metrics = comparator.compare(
        generated_path='generated_table.csv',
        ground_truth_path='ground_truth_table.csv'
    )

    # Generate report
    report = comparator.generate_report(metrics)

    # Print summary
    print(f"Overall Similarity: {metrics.overall_score * 100:.2f}%")
    print(f"Structural Score: {metrics.structural_score * 100:.2f}%")
    print(f"Semantic Score: {metrics.semantic_score * 100:.2f}%")
    print(f"Exact Match Rate: {metrics.exact_match_rate * 100:.2f}%")
    print(f"Mismatched Cells: {len(metrics.mismatched_cells)}")
    print(f"\nQuality: {report['quality_assessment']}")