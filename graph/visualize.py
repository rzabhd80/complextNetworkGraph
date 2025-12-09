import json
import igraph as ig

# --- 1. Initialization ---
tabels = []
g = ig.Graph() # Graph should be directed for containment

# Data collectors for the three types of vertices
vertex_data = [] # To store attributes for all vertices
all_semantic_ids = set() # To ensure unique IDs for all vertices
edges_to_add = []

# --- 2. Load Data ---
try:
    file_path = "" # Define outside for error message access
    for j in range(1, 3):
        for i in range(1, 8):
            file_path = f"../table_extractor/out/table_{j}.json_table{i}_data.json"
            with open(file_path, "r") as file:
                tabels.append(json.load(file))
except FileNotFoundError:
    print(f"Error: One of the files was not found at {file_path}")
except json.JSONDecodeError:
    print(f"Error: File {file_path} contains invalid JSON.")


# --- 3. Iterate and Collect Vertex Data and Edges ---
for i, table in enumerate(tabels):
    # Use a string ID for consistency
    table_semantic_id = f"T_{i + 1}"

    if table_semantic_id not in all_semantic_ids:
        all_semantic_ids.add(table_semantic_id)
        # Collect Table Vertex Data
        vertex_data.append(
            {
                "name": table_semantic_id,
                "type": "Table",
                "bbox": table.get("detected_bbox"),
                "x_offset": table.get("coordinate_space", {}).get("original_image_offset_applied", {}).get("x"),
                "y_offset": table.get("coordinate_space", {}).get("original_image_offset_applied", {}).get("y"),
                "text": None,
            }
        )

    # LINE VERTICES
    lines = table.get("borders", {}).get("horizontal", [])
    for line in lines:
        # Use .get('id') or .get('line_id') for robust ID retrieval
        line_core_id = line.get('id') or line.get('line_id')
        if not line_core_id: continue # Skip if no valid ID found

        line_semantic_id = f"{table_semantic_id}_L_{line_core_id}"

        if line_semantic_id not in all_semantic_ids:
            all_semantic_ids.add(line_semantic_id)
            # Collect Line Vertex Data
            vertex_data.append(
                {
                    "name": line_semantic_id,
                    "type": "Line",
                    "y_coord": line.get("y_coord"),
                    "bbox": None,
                    "x_offset": None,
                    "y_offset": None,
                    "text": None,
                }
            )

        # EDGE: Table -> Line
        edges_to_add.append((table_semantic_id, line_semantic_id))

    # WORD VERTICES
    words = table.get("word_to_border_mappings", [])
    for word in words:
        try:
            word_semantic_id = f"{table_semantic_id}_W_{word.get('word_id')}"

            if word_semantic_id not in all_semantic_ids:
                all_semantic_ids.add(word_semantic_id)
                # Collect Word Vertex Data
                vertex_data.append(
                    {
                        "name": word_semantic_id,
                        "type": "Word",
                        "text": word.get("word_text"),
                        "bbox": word.get("detected_bbox"),
                        "center": word.get("word_center"),
                        "x_offset": None,
                        "y_offset": None,
                    }
                )

            # EDGE: Line -> Word (from top and bottom borders)

            # 1. Top Line connection
            top_border_data = word.get('borders', {}).get('top')
            if top_border_data and top_border_data.get('line_id'):
                top_line_id_core = top_border_data.get('line_id')
                top_line_id = f"{table_semantic_id}_L_{top_line_id_core}"
                edges_to_add.append((top_line_id, word_semantic_id))

            # 2. Bottom Line connection
            bottom_border_data = word.get('borders', {}).get('bottom')
            if bottom_border_data and bottom_border_data.get('line_id'):
                bottom_line_id_core = bottom_border_data.get('line_id')
                bottom_line_id = f"{table_semantic_id}_L_{bottom_line_id_core}"
                edges_to_add.append((bottom_line_id, word_semantic_id))

        except Exception as e:
            # Catch any unexpected errors during word processing
            # print(f"Unexpected error processing word {word.get('word_id')}: {e}")
            continue


# --- 4. Final Graph Construction ---
# Create lists for igraph's attribute dictionary
names = [d["name"] for d in vertex_data]
types = [d["type"] for d in vertex_data]
texts = [d["text"] for d in vertex_data]
bboxes = [d.get("bbox") for d in vertex_data]
x_offsets = [d.get("x_offset") for d in vertex_data]

# Add all vertices at once with their attributes
g.add_vertices(
    len(names),
    attributes={
        "name": names,
        "type": types,
        "text": texts,
        "bbox": bboxes,
        "x_offset": x_offsets,
    },
)

# Map semantic IDs to igraph's internal integer IDs
name_to_id = {name: i for i, name in enumerate(g.vs["name"])}

# Convert the semantic edges to igraph internal indices
unique_indexed_edges = set()
for u, v in edges_to_add:
    if u in name_to_id and v in name_to_id:
        unique_indexed_edges.add((name_to_id[u], name_to_id[v]))

g.add_edges(list(unique_indexed_edges))

# --- 5. Verification and Export ---
print("--- Graph Construction Complete ---")
print(f"Total Vertices: {g.vcount()}")
print(f"Total Edges: {g.ecount()}")

output_file = "table_structure_graph.graphml"

try:
    g.write_graphml(output_file)
    print(f"\n✅ Graph successfully exported to: {output_file}")

except Exception as e:
    print(f"\n❌ Error during GraphML export: {e}")
