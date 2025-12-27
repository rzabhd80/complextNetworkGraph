import json
import igraph as ig
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class TableGraphBuilder:
    """Build and visualize table structure graphs with spatial connections."""

    def __init__(self):
        self.documents_data = []

    def load_json_files(self, output_dir: str = "./out") -> List[Dict[str, Any]]:
        """Load ALL *_doc.json files from output directory."""
        output_path = Path(output_dir)

        if not output_path.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")

        doc_files = sorted(output_path.glob("*_doc.json"))

        if not doc_files:
            raise FileNotFoundError(f"No *_doc.json files found in {output_dir}")

        print(f"Found {len(doc_files)} document files:")

        documents = []
        for doc_file in doc_files:
            print(f"  - {doc_file.name}")
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    documents.append(json.load(f))
            except json.JSONDecodeError as e:
                print(f"    ⚠ Error reading {doc_file.name}: {e}")
                continue

        print(f"\n✓ Successfully loaded {len(documents)} documents\n")
        self.documents_data = documents
        return documents

    def unify_coordinates(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """Convert relative coordinates to absolute image space."""
        table_x1, table_y1 = table["bbox"][0], table["bbox"][1]

        for line in table.get("lines", []):
            line["bbox_absolute"] = [
                line["bbox"][0] + table_x1,
                line["bbox"][1] + table_y1,
                line["bbox"][2] + table_x1,
                line["bbox"][3] + table_y1,
            ]

            line["line_cord_absolute"] = [
                [pt[0] + table_x1, pt[1] + table_y1]
                for pt in line["line_cord"]
            ]

            for word in line.get("words", []):
                word["cordinate_absolute"] = [
                    [pt[0] + table_x1, pt[1] + table_y1]
                    for pt in word["cordinate"]
                ]

        return table

    def build_single_document_graph(self, doc_data: Dict[str, Any]) -> ig.Graph:
        """Build graph for a single document with spatial table connections."""
        g = ig.Graph(directed=True)
        vertex_data = []
        all_semantic_ids: Set[str] = set()
        edges_to_add: List[Tuple[str, str, str]] = []  # (source, target, edge_type)

        doc_id = doc_data["doc_id"]

        # Add document node
        doc_semantic_id = f"DOC_{doc_id}"
        all_semantic_ids.add(doc_semantic_id)
        vertex_data.append({
            "name": doc_semantic_id,
            "type": "Document",
            "label": f"Doc {doc_id}",
            "bbox": None,
            "text": None,
            "conf": None,
        })

        # Process tables
        for table in doc_data.get("tables", []):
            table_id = table["id"]
            table_semantic_id = f"DOC_{doc_id}_TABLE_{table_id}"

            if table_semantic_id not in all_semantic_ids:
                all_semantic_ids.add(table_semantic_id)
                table = self.unify_coordinates(table)

                vertex_data.append({
                    "name": table_semantic_id,
                    "type": "Table",
                    "label": f"T{table_id}",
                    "bbox": str(table["bbox"]),
                    "text": None,
                    "conf": table.get("confidence"),
                })

            # Edge: Document -> Table (hierarchical)
            edges_to_add.append((doc_semantic_id, table_semantic_id, "hierarchy"))

            # Add spatial table-to-table connections based on distances
            for other_table_id, distance_info in table.get("distances", {}).items():
                other_table_semantic_id = f"DOC_{doc_id}_TABLE_{other_table_id}"
                # Add spatial proximity edge
                edges_to_add.append((
                    table_semantic_id,
                    other_table_semantic_id,
                    "spatial"
                ))

            # Process lines
            for line in table.get("lines", []):
                line_id = line["line_id"]
                line_semantic_id = f"{table_semantic_id}_LINE_{line_id}"

                if line_semantic_id not in all_semantic_ids:
                    all_semantic_ids.add(line_semantic_id)

                    vertex_data.append({
                        "name": line_semantic_id,
                        "type": "Line",
                        "label": line["text"][:20] + ("..." if len(line["text"]) > 20 else ""),
                        "bbox": str(line.get("bbox_absolute", line["bbox"])),
                        "text": line["text"],
                        "conf": line.get("conf"),
                    })

                # Edge: Table -> Line (hierarchical)
                edges_to_add.append((table_semantic_id, line_semantic_id, "hierarchy"))

                # Process words
                for word_idx, word in enumerate(line.get("words", [])):
                    word_semantic_id = f"{line_semantic_id}_WORD_{word_idx}"

                    if word_semantic_id not in all_semantic_ids:
                        all_semantic_ids.add(word_semantic_id)

                        coords = word.get("cordinate_absolute", word["cordinate"])
                        xs = [pt[0] for pt in coords]
                        ys = [pt[1] for pt in coords]
                        bbox = [min(xs), min(ys), max(xs), max(ys)]

                        vertex_data.append({
                            "name": word_semantic_id,
                            "type": "Word",
                            "label": word["word"][:10],
                            "bbox": str(bbox),
                            "text": word["word"],
                            "conf": None,
                        })

                    # Edge: Line -> Word (hierarchical)
                    edges_to_add.append((line_semantic_id, word_semantic_id, "hierarchy"))

        # Build igraph
        names = [d["name"] for d in vertex_data]
        types = [d["type"] for d in vertex_data]
        labels = [d["label"] for d in vertex_data]
        bboxes = [d.get("bbox") for d in vertex_data]
        texts = [d.get("text") for d in vertex_data]
        confs = [d.get("conf") for d in vertex_data]

        g.add_vertices(
            len(names),
            attributes={
                "name": names,
                "type": types,
                "label": labels,
                "bbox": bboxes,
                "text": texts,
                "confidence": confs,
            }
        )

        name_to_id = {name: i for i, name in enumerate(g.vs["name"])}

        # Add edges with type information
        unique_indexed_edges = {}  # (source, target) -> edge_type
        for u, v, edge_type in edges_to_add:
            if u in name_to_id and v in name_to_id:
                edge_key = (name_to_id[u], name_to_id[v])
                unique_indexed_edges[edge_key] = edge_type

        g.add_edges(list(unique_indexed_edges.keys()))

        # Add edge type as attribute
        edge_types = [unique_indexed_edges[e] for e in unique_indexed_edges.keys()]
        g.es["edge_type"] = edge_types

        return g

    def build_combined_graph(self) -> ig.Graph:
        """Build a single graph containing ALL documents with spatial connections."""
        g = ig.Graph(directed=True)
        vertex_data = []
        all_semantic_ids: Set[str] = set()
        edges_to_add: List[Tuple[str, str, str]] = []  # (source, target, edge_type)

        # Process all documents
        for doc_data in self.documents_data:
            doc_id = doc_data["doc_id"]

            # Add document node
            doc_semantic_id = f"DOC_{doc_id}"
            if doc_semantic_id not in all_semantic_ids:
                all_semantic_ids.add(doc_semantic_id)
                vertex_data.append({
                    "name": doc_semantic_id,
                    "type": "Document",
                    "label": f"Doc {doc_id}",
                    "bbox": None,
                    "text": None,
                    "conf": None,
                })

            # Process tables
            for table in doc_data.get("tables", []):
                table_id = table["id"]
                table_semantic_id = f"DOC_{doc_id}_TABLE_{table_id}"

                if table_semantic_id not in all_semantic_ids:
                    all_semantic_ids.add(table_semantic_id)
                    table = self.unify_coordinates(table)

                    vertex_data.append({
                        "name": table_semantic_id,
                        "type": "Table",
                        "label": f"D{doc_id}T{table_id}",
                        "bbox": str(table["bbox"]),
                        "text": None,
                        "conf": table.get("confidence"),
                    })

                # Edge: Document -> Table (hierarchical)
                edges_to_add.append((doc_semantic_id, table_semantic_id, "hierarchy"))

                # Add spatial table-to-table connections within same document
                for other_table_id, distance_info in table.get("distances", {}).items():
                    other_table_semantic_id = f"DOC_{doc_id}_TABLE_{other_table_id}"
                    edges_to_add.append((
                        table_semantic_id,
                        other_table_semantic_id,
                        "spatial"
                    ))

                # Process lines
                for line in table.get("lines", []):
                    line_id = line["line_id"]
                    line_semantic_id = f"{table_semantic_id}_LINE_{line_id}"

                    if line_semantic_id not in all_semantic_ids:
                        all_semantic_ids.add(line_semantic_id)

                        vertex_data.append({
                            "name": line_semantic_id,
                            "type": "Line",
                            "label": line["text"][:15] + ("..." if len(line["text"]) > 15 else ""),
                            "bbox": str(line.get("bbox_absolute", line["bbox"])),
                            "text": line["text"],
                            "conf": line.get("conf"),
                        })

                    edges_to_add.append((table_semantic_id, line_semantic_id, "hierarchy"))

                    # Process words
                    for word_idx, word in enumerate(line.get("words", [])):
                        word_semantic_id = f"{line_semantic_id}_WORD_{word_idx}"

                        if word_semantic_id not in all_semantic_ids:
                            all_semantic_ids.add(word_semantic_id)

                            coords = word.get("cordinate_absolute", word["cordinate"])
                            xs = [pt[0] for pt in coords]
                            ys = [pt[1] for pt in coords]
                            bbox = [min(xs), min(ys), max(xs), max(ys)]

                            vertex_data.append({
                                "name": word_semantic_id,
                                "type": "Word",
                                "label": word["word"][:8],
                                "bbox": str(bbox),
                                "text": word["word"],
                                "conf": None,
                            })

                        edges_to_add.append((line_semantic_id, word_semantic_id, "hierarchy"))

        # Build igraph
        names = [d["name"] for d in vertex_data]
        types = [d["type"] for d in vertex_data]
        labels = [d["label"] for d in vertex_data]
        bboxes = [d.get("bbox") for d in vertex_data]
        texts = [d.get("text") for d in vertex_data]
        confs = [d.get("conf") for d in vertex_data]

        g.add_vertices(
            len(names),
            attributes={
                "name": names,
                "type": types,
                "label": labels,
                "bbox": bboxes,
                "text": texts,
                "confidence": confs,
            }
        )

        name_to_id = {name: i for i, name in enumerate(g.vs["name"])}

        # Add edges with type information
        unique_indexed_edges = {}  # (source, target) -> edge_type
        for u, v, edge_type in edges_to_add:
            if u in name_to_id and v in name_to_id:
                edge_key = (name_to_id[u], name_to_id[v])
                unique_indexed_edges[edge_key] = edge_type

        g.add_edges(list(unique_indexed_edges.keys()))

        # Add edge type as attribute
        edge_types = [unique_indexed_edges[e] for e in unique_indexed_edges.keys()]
        g.es["edge_type"] = edge_types

        return g

    def visualize_graph(self, g: ig.Graph, output_file: str, title: str, layout: str = "fr"):
        """Visualize a graph and save as PNG."""

        # Print statistics
        type_counts = {}
        for v in g.vs:
            vtype = v["type"]
            type_counts[vtype] = type_counts.get(vtype, 0) + 1

        print(f"\n{title} Statistics:")
        for vtype, count in sorted(type_counts.items()):
            print(f"  {vtype:12s}: {count:5d}")
        print(f"  Total Vertices: {g.vcount()}")
        print(f"  Total Edges: {g.ecount()}")

        # Count edge types
        if g.ecount() > 0 and "edge_type" in g.es.attributes():
            edge_type_counts = {}
            for edge in g.es:
                etype = edge["edge_type"] if "edge_type" in edge.attributes() else "unknown"
                edge_type_counts[etype] = edge_type_counts.get(etype, 0) + 1
            print(f"  Edge Types:")
            for etype, count in sorted(edge_type_counts.items()):
                print(f"    {etype:12s}: {count:5d}")

        # Check connectivity
        components = g.components(mode='weak')
        print(f"  Connected Components: {len(components)}")
        print(f"  Is Connected: {len(components) == 1}")
        print(f"  Is DAG: {g.is_dag()}")

        # Color and size mapping
        color_map = {
            "Document": "#3b82f6",
            "Table": "#22c55e",
            "Line": "#f59e0b",
            "Word": "#a855f7"
        }

        size_map = {
            "Document": 50,
            "Table": 40,
            "Line": 25,
            "Word": 15
        }

        vertex_colors = [color_map.get(vtype, "#gray") for vtype in g.vs["type"]]
        vertex_sizes = [size_map.get(vtype, 20) for vtype in g.vs["type"]]

        # Edge colors based on type
        edge_colors = []
        if g.ecount() > 0 and "edge_type" in g.es.attributes():
            for edge in g.es:
                etype = edge["edge_type"] if "edge_type" in edge.attributes() else "hierarchy"
                if etype == "spatial":
                    edge_colors.append("#ff0000")  # Red for spatial connections
                else:
                    edge_colors.append("#808080")  # Gray for hierarchy
        else:
            edge_colors = ["#808080"] * g.ecount()  # Default to gray

        # Layout
        if layout == "tree":
            roots = [v.index for v in g.vs if v["type"] == "Document"]
            if roots:
                layout_obj = g.layout_reingold_tilford(root=roots)
            else:
                layout_obj = g.layout_fruchterman_reingold()
        elif layout == "fr":
            layout_obj = g.layout_fruchterman_reingold(niter=500)
        elif layout == "kk":
            layout_obj = g.layout_kamada_kawai()
        elif layout == "circle":
            layout_obj = g.layout_circle()
        elif layout == "drl":
            layout_obj = g.layout_drl()
        else:
            layout_obj = g.layout_auto()

        # Create visual style
        edge_widths = []
        if g.ecount() > 0 and "edge_type" in g.es.attributes():
            for edge in g.es:
                etype = edge["edge_type"] if "edge_type" in edge.attributes() else "hierarchy"
                edge_widths.append(2 if etype == "spatial" else 1)
        else:
            edge_widths = [1] * g.ecount()

        visual_style = {
            "vertex_size": vertex_sizes,
            "vertex_color": vertex_colors,
            "vertex_label": g.vs["label"],
            "vertex_label_size": 7,
            "edge_color": edge_colors,
            "edge_arrow_size": 0.5,
            "edge_arrow_width": 0.5,
            "edge_width": edge_widths,
            "layout": layout_obj,
            "bbox": (2400, 1800),
            "margin": 80
        }

        # Save PNG
        print(f"\nGenerating visualization with '{layout}' layout...")
        try:
            ig.plot(g, output_file, **visual_style)
            print(f"  ✓ Visualization saved to: {output_file}")
        except Exception as e:
            print(f"  ✗ Error creating visualization: {e}")

    def visualize_all_documents(self, output_dir: str = "./out", layout: str = "fr"):
        """Visualize each document separately."""
        if not self.documents_data:
            print("No documents loaded. Please run load_json_files() first.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"VISUALIZING {len(self.documents_data)} DOCUMENTS SEPARATELY")
        print(f"{'=' * 60}")

        for doc_data in self.documents_data:
            doc_id = doc_data["doc_id"]
            print(f"\n{'=' * 60}")
            print(f"Processing Document {doc_id}")
            print(f"{'=' * 60}")

            g = self.build_single_document_graph(doc_data)
            output_file = str(output_path / f"doc_{doc_id}_graph.png")
            self.visualize_graph(g, output_file, f"Document {doc_id}", layout)

    def visualize_combined(self, output_dir: str = "./out", layout: str = "fr"):
        """Visualize all documents in a single graph."""
        if not self.documents_data:
            print("No documents loaded. Please run load_json_files() first.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"CREATING COMBINED GRAPH (ALL DOCUMENTS)")
        print(f"{'=' * 60}")

        g = self.build_combined_graph()
        output_file = str(output_path / "combined_all_documents.png")
        self.visualize_graph(g, output_file, "Combined Graph (All Documents)", layout)

    def analyze_degree_distribution(self, g: ig.Graph, title: str, output_file: str):
        """Analyze and plot degree distribution in log-log scale."""
        from collections import Counter

        # Configure matplotlib for Persian text
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

        # Get degree sequence
        degrees = g.degree()

        # Count degree frequencies
        degree_counts = Counter(degrees)

        # Prepare data for plotting
        k_values = sorted(degree_counts.keys())
        p_k = [degree_counts[k] / len(degrees) for k in k_values]

        # Filter for log-log plots (remove zeros)
        filtered_k = [k for k in k_values if k > 0 and degree_counts[k] > 0]
        filtered_p = [degree_counts[k] / len(degrees) for k in filtered_k]

        # Check if we have enough data for log plots
        has_positive_data = len(filtered_k) > 0

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f'analysis of degree distribution - {title}', fontsize=18, fontweight='bold')

        # Plot 1: Regular histogram
        ax1 = axes[0, 0]
        ax1.bar(k_values, p_k, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Degree (k)', fontsize=13)
        ax1.set_ylabel('P(k)', fontsize=13)
        ax1.set_title('degree distribution in standard scale', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Log-Y scale
        ax2 = axes[0, 1]
        if has_positive_data:
            ax2.bar(filtered_k, [degree_counts[k] / len(degrees) for k in filtered_k],
                    color='darkgreen', alpha=0.7, edgecolor='black')
            ax2.set_yscale('log')
            ax2.set_xlabel('Degree (k)', fontsize=13)
            ax2.set_ylabel('P(k)', fontsize=13)
            ax2.set_title('degree distribution in semi log scale', fontsize=14)
            ax2.grid(True, alpha=0.3, which='both')
        else:
            ax2.text(0.5, 0.5, 'No positive degree values',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('degree distribution in log scale', fontsize=14)

        # Plot 3: Log-Log scale (scatter)
        ax3 = axes[1, 0]
        if has_positive_data and len(filtered_k) > 1:
            ax3.scatter(filtered_k, filtered_p, color='red', alpha=0.6, s=60,
                        edgecolors='darkred', linewidths=1.5)
            ax3.set_xscale('log')
            ax3.set_yscale('log')
            ax3.set_xlabel('log(k)', fontsize=13)
            ax3.set_ylabel('log(P(k))', fontsize=13)
            ax3.set_title('degree distribution in log-log', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, which='both')

            # Try to fit a line for power-law estimation
            if len(filtered_k) > 2:
                log_k = np.log10(filtered_k)
                log_p = np.log10(filtered_p)
                coeffs = np.polyfit(log_k, log_p, 1)
                fit_line = np.poly1d(coeffs)
                ax3.plot(filtered_k, 10 ** (fit_line(log_k)), 'b--', linewidth=2.5,
                         label=f'power law: γ ≈ {-coeffs[0]:.2f}')
                ax3.legend(fontsize=11)
        else:
            ax3.text(0.5, 0.5, 'Insufficient data for log-log plot',
                     ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('degree distribution in log-log', fontsize=14, fontweight='bold')

        # Plot 4: Cumulative distribution (CCDF) in log-log scale
        ax4 = axes[1, 1]
        sorted_degrees = sorted([d for d in degrees if d > 0], reverse=True)

        if len(sorted_degrees) > 0:
            ccdf_x = []
            ccdf_y = []
            for k in sorted(set(sorted_degrees)):
                if k > 0:
                    ccdf_x.append(k)
                    ccdf_y.append(sum(1 for d in degrees if d >= k) / len(degrees))

            if len(ccdf_x) > 1:
                ax4.scatter(ccdf_x, ccdf_y, color='purple', alpha=0.6, s=60,
                            edgecolors='indigo', linewidths=1.5)
                ax4.set_xscale('log')
                ax4.set_yscale('log')
                ax4.set_xlabel('log(k)', fontsize=13)
                ax4.set_ylabel('log(P(K≥k))', fontsize=13)
                ax4.set_title(' distribution in accumulative log-log', fontsize=14)
                ax4.grid(True, alpha=0.3, which='both')

                if len(ccdf_x) > 2:
                    log_ccdf_x = np.log10(ccdf_x)
                    log_ccdf_y = np.log10(ccdf_y)
                    coeffs_ccdf = np.polyfit(log_ccdf_x, log_ccdf_y, 1)
                    fit_line_ccdf = np.poly1d(coeffs_ccdf)
                    ax4.plot(ccdf_x, 10 ** (fit_line_ccdf(log_ccdf_x)), 'b--',
                             linewidth=2.5, label=f'power law: γ ≈ {-coeffs_ccdf[0] + 1:.2f}')
                    ax4.legend(fontsize=11)
            else:
                ax4.text(0.5, 0.5, 'Insufficient data for CCDF',
                         ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title(' distribution in accumulative log-log', fontsize=14)
        else:
            ax4.text(0.5, 0.5, 'No positive degree values',
                     ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('degree distribution in accumulative log-log', fontsize=14)

        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()

        # Print statistics
        print(f"\n{'=' * 60}")
        print(f"Degree Distribution Statistics - {title}")
        print(f"{'=' * 60}")
        print(f"  Mean degree: {np.mean(degrees):.2f}")
        print(f"  Median degree: {np.median(degrees):.2f}")
        print(f"  Min degree: {min(degrees)}")
        print(f"  Max degree: {max(degrees)}")
        print(f"  Std deviation: {np.std(degrees):.2f}")
        print(f"  Unique degree values: {len(set(degrees))}")

        print(f"\n  Average degree by node type:")
        for node_type in set(g.vs["type"]):
            type_degrees = [g.degree(v) for v in g.vs if v["type"] == node_type]
            if type_degrees:
                print(f"    {node_type:12s}: {np.mean(type_degrees):.2f}")

        print(f"{'=' * 60}\n")

        return degree_counts

    def analyze_all_degree_distributions(self, output_dir: str = "./out"):
        """Analyze degree distributions for all graphs."""
        if not self.documents_data:
            print("No documents loaded.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"degree analysis for all graphs")
        print(f"{'=' * 60}")

        # Analyze individual documents
        for doc_data in self.documents_data:
            doc_id = doc_data["doc_id"]
            g = self.build_single_document_graph(doc_data)
            output_file = str(output_path / f"degree_dist_doc_{doc_id}.png")
            self.analyze_degree_distribution(g, f"doc {doc_id}", output_file)
            print(f"  ✓ doc analysis {doc_id} stored: {output_file}")

        # Analyze combined graph
        g_combined = self.build_combined_graph()
        output_file = str(output_path / "degree_dist_combined.png")
        self.analyze_degree_distribution(g_combined, "combined graph of all docs", output_file)
        print(f"  ✓ combined graph analysis stored: {output_file}")

    def _create_legend(self, output_dir: str = "./out"):
        """Create a legend for the graph visualizations."""
        color_map = {
            "Document": "#3b82f6",
            "Table": "#22c55e",
            "Line": "#f59e0b",
            "Word": "#a855f7"
        }

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.axis('off')

        patches = [
            mpatches.Patch(color=color, label=node_type)
            for node_type, color in color_map.items()
        ]

        # Add edge type legend
        patches.append(mpatches.Patch(color="#808080", label="Hierarchy Edge"))
        patches.append(mpatches.Patch(color="#ff0000", label="Spatial Edge"))

        ax.legend(handles=patches, loc='center', fontsize=12, title="Legend")
        plt.tight_layout()

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        legend_file = output_path / "graph_legend.png"

        plt.savefig(legend_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n  ✓ Legend saved to: {legend_file}")


def visualize():
    print("=" * 60)
    print("TABLE STRUCTURE GRAPH BUILDER WITH SPATIAL CONNECTIONS")
    print("=" * 60)

    builder = TableGraphBuilder()

    # Load documents
    builder.load_json_files("./out")

    # Visualize each document separately (with spatial table connections within each page)
    builder.visualize_all_documents(output_dir="./out", layout="fr")

    # Visualize combined graph (all documents in one visualization)
    builder.visualize_combined(output_dir="./out", layout="fr")

    # Analyze degree distributions
    builder.analyze_all_degree_distributions(output_dir="./out")

    # Create legend
    builder._create_legend(output_dir="./out")

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print("\nOutput files:")
    print("  - doc_N_graph.png: Individual document visualizations")
    print("  - combined_all_documents.png: All documents in one graph")
    print("  - degree_dist_doc_N.png: Degree distribution analysis per document")
    print("  - degree_dist_combined.png: Degree distribution for combined graph")
    print("  - graph_legend.png: Color/edge type legend")
    print("\nEdge Types:")
    print("  - Gray edges: Hierarchical relationships (Doc→Table→Line→Word)")
    print("  - Red edges: Spatial proximity between tables on same page")
    print("=" * 60)
