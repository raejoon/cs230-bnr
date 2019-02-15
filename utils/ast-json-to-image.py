#!/usr/bin/python3

""" Module to visualize abstract syntax trees serialized in json.
    The following packages need to be additionally installed.
    1. networkx
    2. matplotlib
    3. graphviz
    4. pydot
"""

import json
import matplotlib.pyplot as plt
import networkx as nx
import os.path

def parse_json(json_file):
    """ Deserializes the json string in a file and returns the object.

    Refer to the tranaslation table in the Python documentation for json
    library.
    """
    with open(json_file) as fo:
        json_string = fo.read()
    parsed_json = json.loads(json_string)
    return parsed_json

def construct_nx_tree(json_tree):
    """ Returns a networkx directed graph object based on the tree expressed by
    the JSON object.

    The json-decoded object representation represents a node with a dictionary
    of its attributes. The key values are {"id", "type", children"}.
    One of the key, value pairs is ("children", list of nodes). The entire tree
    is represented in a nested manner, dictionary->list->dictionary->...
    
    Each node in the networkx directed graph has attributes, {"id", "type"}.
    The values are copied from the json object.
    
    Parameters:
    json_tree (JSON object): json-decoded object representation of a tree

    Returns:
    networkx.DiGraph: Directed graph representation of a tree
    """
    G = nx.DiGraph()
    json_node = json_tree
    G.add_node(json_node["id"], nodetype=json_node["type"])

    dfs_stack = [json_node]
    while len(dfs_stack) > 0:
        json_node = dfs_stack.pop()
        if "children" not in json_node:   # leaf node
            continue

        for child_node in json_node["children"]:
            if child_node["id"] not in G:
                G.add_node(child_node["id"], nodetype=child_node["type"])
            G.add_edge(json_node["id"], child_node["id"])
            dfs_stack.append(child_node)
    return G

def draw_tree(nx_graph_tree, title=None):
    """ Draws a tree expressed by a networkx directed graph object.

    The graph must be acyclic and have no partitions in order to represent a
    tree.

    Parameters:
    nx_graph_tree (nx.Graph): Directed graph object. 
    title (string): Title of the figure
    """
    G = nx_graph_tree
    if title is not None:
        plt.title(title, fontsize=18)
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    labels = {n: "%s\n%s" % (n, d["nodetype"]) for n,d in G.nodes(data=True)}
    nx.draw(G, pos, arrows=True, with_labels=True, labels=labels,
            node_color="None", font_size=14, node_size=800)
    plt.tight_layout()

def generate_ast_image(json_file):
    """ Generates a image file of the abstract syntax tree from a json file.
    
    Generated image path is ./images/(basename of json file).png.
    Throws an error if ./images directory do no exist.
    """
    image_name = os.path.join("images/",
                    os.path.splitext(os.path.basename(json_file))[0] + ".png")

    if not os.path.exists(image_name):
        fig = plt.figure(figsize=(16, 9))
        parsed_json = parse_json(json_file)
        tree = construct_nx_tree(parsed_json)
        draw_tree(tree, title="AST: " + str(json_file))
        plt.savefig(image_name)
        plt.close()
        print(json_file, "created.")
    else:
        print(json_file, "already there.")

if __name__=="__main__":
    """ Search all json files under the current directory (recursively) and
    generates the abstract syntax tree images. 
    """
    import sys, subprocess, multiprocessing
    find_sp = subprocess.run(["find", "asts", "-name", "*.json"],
                stdout=subprocess.PIPE)
    filename_list = find_sp.stdout.decode("utf-8").split()

    with multiprocessing.Pool(8, maxtasksperchild=50) as p:
        p.map(generate_ast_image, filename_list) 
