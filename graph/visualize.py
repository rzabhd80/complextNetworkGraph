import json

import igraph as ig

tabels = []
g = ig.Graph()
table_ids = set()
line_ids = set()
word_ids = set()

edges_to_add = []
try:
    for j in range(1, 3):
        for i in range(1, 8):
            with open(
                f"../table_extractor/out/table_{j}.json_table{i}_data.json", "r"
            ) as file:
                tabels.append(json.load(file))
except FileNotFoundError:
    print("Error: The file was not found.")
except json.JSONDecodeError:
    print("Error: The file contains invalid JSON.")
for i, table in enumerate(tabels):
    table_id = i + 1
    table_ids.add(table_id)
    g.add_vertices(table_id)
    x = table.get("coordinate_space").get("original_image_offset_applied").get("x")
    y = table.get("coordinate_space").get("original_image_offset_applied").get("y")

    tabel_box = table.get("detected_bbox")
    words = table.get("word_to_border_mappings")
    lines = table.get("borders").get("horizontal")
    for line in lines:
        line_id = f"table{table_id}_" + line.get("line_id")
        line_ids.add(line_id)

        edges_to_add.append((table_id, line_id))
    for word in words:
        try:
            word_id = f"table{table_id}_" + word.get("word_id")
            word_text = word.get("word_text")
            word_center = word.get("word_center")
            if word.get("borders").get("top", None):
                top_line_id = f"table{table_id}_" + word.get("borders").get("top").get(
                    "line_id"
                )
            if word.get("borders").get("bottom", None):
                bottom_line_id = f"table{table_id}_" + word.get("borders").get(
                    "bottom"
                ).get("line_id")
            word_ids.add(word_id)

            edges_to_add.append((top_line_id, word_id))
            edges_to_add.append((top_line_id, word_id))
        except:
            print(f"{word_id}_{table_id}")
            print(word)
all_ids = list(table_ids) + list(line_ids) + list(word_ids)
g.add_vertices(len(all_ids))
g.vs["name"] = all_ids
name_to_id = {name: i for i, name in enumerate(all_ids)}
indexed_edges = [(name_to_id[u], name_to_id[v]) for u, v in edges_to_add]
g.add_edges(indexed_edges)
print(f"Total Vertices: {g.vcount()}")
print(f"Total Edges: {g.ecount()}")
print("\nVertex Names (IDs):")
print(g.vs["name"])
print("\nEdges (Source ID -> Target ID):")
print(g.get_edgelist())

# Example: Check neighbors of the Table 'T001'
# table_vertex = g.vs.find(name="T001")
# print(
# f"\nNeighbors of Table T001 (Lines contained): {g.neighbors(table_vertex, mode='out')}"
# )
