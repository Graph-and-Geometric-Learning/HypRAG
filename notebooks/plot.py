import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import networkx as nx

# Parse XML
def parse_bubble(bubble_element, parent=None, graph=None):
    name = bubble_element.attrib.get('name', '').strip()
    node_id = id(bubble_element)
    label = name if name else f"Bubble_{node_id}"
    graph.add_node(node_id, label=label)
    if parent:
        graph.add_edge(parent, node_id)

    for child in bubble_element:
        if child.tag == 'Nugget':
            nugget_id = f"Nugget_{child.attrib['id']}"
            graph.add_node(nugget_id, label=nugget_id)
            graph.add_edge(node_id, nugget_id)
        elif child.tag == 'Bubble':
            parse_bubble(child, node_id, graph)

    return graph

# Load and parse XML string
def build_graph_from_xml(xml_string):
    root = ET.fromstring(xml_string)
    G = nx.DiGraph()
    for bubble in root.findall('Bubble'):
        parse_bubble(bubble, parent=None, graph=G)
    return G

# Plot graph
def plot_graph(G, output_file="graph.png"):
    plt.figure(figsize=(20, 15))
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=False, node_size=50, edge_color='gray', alpha=0.6)
    nx.draw_networkx_labels(G, pos, labels, font_size=6)
    plt.title("Bubble and Nugget Graph")
    plt.axis('off')
    plt.savefig(output_file, dpi=300)
    plt.close()

# Usage
with open("input.xml", "r", encoding="utf-8") as f:
    xml_data = f.read()

G = build_graph_from_xml(xml_data)
plot_graph(G)
