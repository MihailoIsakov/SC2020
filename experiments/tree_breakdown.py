"""
This script partitions the HDBSCAN minimum spanning tree from the top down,
iteratively splitting the tree at the longest edges, and tries to interpret
the split. We do this in two ways:
    1. for each split, we train a decision tree to predict where each job in the
    original tree should go. This tree can tell us more about the split.
    2. for each split, we train predictors and evaluate how good our predictions 
    are. If the predictions are good for e.g., linear models, we know that there 
    is very little variance in the rest of the tree and that we can stop.
"""
import sys
import copy
import numpy as np
import networkx as nx
import hdbscan
from joblib import Memory
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Import root module
sys.path.insert(0, '../')
import dataset

# Plotting
import matplotlib
import matplotlib.pyplot as plt
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                          "PyGraphviz or pydot")

# This will allow memoization:
memory = Memory(location='.cache', verbose=0)


new_node_idx = 0


def condensed_tree_to_linkage_matrix(ct):
    """
    We can get a condensed tree from HDBSCAN, but scipy's dendogram (scipy.cluster.hierarchy.dendrogram) 
    requires us to pass linkage matrices. Condensed tree matrices are Nx4 matrices of edges, where in a row
    (X, Y, Z, W):
        * X is the parent node
        * Y is the child node
        * Z is the lambda value specifying at which lambda does the child separate from the parent
        * W is the size of the child, 1 if it a leaf, or k if it is a cluster.
    
    Linkage matrices are Nx4 matrices where each row represents the connection of two leaves or clusters. 
    In a row (X, Y, Z, W):
        * X and Y are the indexes of either leaf nodes or branch (cluster) nodes. Both X and Y are of the same 'level'.
        * Z specifies the 'height' at which the nodes are connected.
        * W is the number of children of this new leaf node (it seems it is not used).

    Both these data structures are similar. The main difference is in the first and the last column.
    In the linkage matrices, columns X and Y are equal, and the resulting branch receives a new index. 
    In condensed tree matrices however, the first column is the parent column, and a parent can have multiple
    (more than 2) children. 
    """
    pass


def split_node(G, parent):
    """
    Takes a node with 3 or more edges, splits it into multiple nodes of degree 2.
    E.g.: 

    G                G
  g |   w          g |   w
    P-------         M------
    |\     |       w |     |
    | \    |   --\   P---  |
    |  \   |   --/   |  |  |
    A   B  C         A  B  C
    """
    global new_node_idx

    while G.out_degree(parent) > 2: 
        try: 
            grandparent = list(G.predecessors(parent))[0]
        except:
            return 
        connected = list(G[parent]) 
        weights   = [G[parent][c]['weight'] for c in connected]
        child     = connected[np.argmin(weights)] # first child 

        # Now delete the grandparent-parend and parent-child edges
        gp_weight = G[grandparent][parent]['weight']
        pc_weight = G[parent][child]['weight']
        G.remove_edges_from([(grandparent, parent), (parent, child)])

        # Now add middle, and connect G-M, M-P, M-C
        middle = "{}-{}".format(str(grandparent).split('-')[0], new_node_idx)
        new_node_idx += 1
        assert middle not in G
        G.add_node(middle)
        G.add_weighted_edges_from([(grandparent, middle, gp_weight), (middle, parent, pc_weight), (middle, child, pc_weight)])


def split_multidegree_nodes(G):
    """
    Takes a graph G and splits any nodes that have more than 2 edges leading out from it.
    The edges with the largest lambdas should be processed first.
    """
    while np.max([v for k, v in G.out_degree]) > 2: 
        for parent in list(G.nodes):
            split_node(G, parent)

    assert np.max([v for k, v in G.out_degree]) <= 2

    # since the inserted nodes don't have sizes, this should populate them
    root = [n for n, d in G.in_degree() if d==0][0]
    # Sometime the graph is too deep? 
    sys.setrecursionlimit(10000)
    populate_node_sizes(G, root)


def get_leaves(G): 
    return [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]


def populate_node_sizes(G, node):
    """
    Nodes in G should have sizes, but some don't after we split multidegree nodes. This
    recursively recalculates sizes of all nodes in the graph starting from node.
    """
    children = list(G[node])
    # if a leaf node, set size to 1
    if len(children) == 0:
        size = 1
    # branches add up children's sizes 
    elif len(children) == 2:
        size = populate_node_sizes(G, children[0]) + populate_node_sizes(G, children[1])
    else:
        raise RuntimeError("Node {} has {} children.".format(node, len(children)))

    G.nodes[node]['size'] = size

    return size


def get_DT_for_node(df, G, node, column_names=None, max_depth=1):
    """
    Finds which leaves belong to each of the children of the node, and trains a dataset to predict these.
    Must be applied on the original graph.
    """
    column_names = copy.deepcopy(column_names)
    left, right = list(G[node])

    left_leaves  = get_leaves(nx.dfs_tree(G, left))
    right_leaves = get_leaves(nx.dfs_tree(G, right))

    # Build a dataset
    df_subset = df.loc[left_leaves + right_leaves]
    y = [1] * len(left_leaves) + [0] * len(right_leaves) 

    # In a loop, keep testing out different trees until accuracy drops
    while len(column_names) > 0:
        # Train DF
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(df_subset, y)

        # Evaluate on train set - the tree is simple enough to not risk overfitting 
        acc = accuracy_score(y, clf.predict(df_subset))

        # Return a text representation
        text = sklearn.tree.export_text(clf, feature_names=column_names)
        text = "Accuracy: {:.2f}\n".format(acc) + text
         
        # Sometimes DT produces no explanation? 
        if len(text.splitlines()) == 2: 
            break
        
        # Remove the best feature
        for idx, column in enumerate(column_names):
            if column in text:
                df_subset = df_subset.drop(columns=column)
                del column_names[idx]
                
        print("Analyzing cluster {} with {} jobs".format(node, df_subset.shape[0]))
        print(text)

        if acc < 0.9:
            break


def _merge_chain_nodes(G, node, dont_merge=[]):
    """
    Traverses the graph, and converts chains such as A-B-C to A-C or A-B to B.
    Dont merge specifies nodes that should not be merged even if part of chain
    """
    if len(list(G.predecessors(node))) == 1 and len(list(G.successors(node))) == 1 and node not in dont_merge:
        parent = list(G.predecessors(node))[0]
        child  = list(G.successors  (node))[0]
        weight = G[node][child]['weight']

        G.remove_node(node)
        G.add_weighted_edges_from([(parent, child, weight)])

        _merge_chain_nodes(G, child)
    else:
        for child in list(G.successors(node)):
            _merge_chain_nodes(G, child)


def build_condensed_graph(G, min_epsilon, min_cluster_size, dont_merge=[]):
    """ 
    Finds nodes in the graph that have edges weight weights above min_epsilon,
    and both children have a size larger than min_cluster_size.
    """
    def filter_node(n):
        return G.nodes[n]['size'] > min_cluster_size

    def filter_edge(n1, n2):
        return 1 / G[n1][n2]['weight'] > min_epsilon

    SG = nx.subgraph_view(G, filter_node=filter_node, filter_edge=filter_edge)
    SG = nx.DiGraph(SG)

    root = [n for n, d in SG.in_degree() if d==0][0]
    _merge_chain_nodes(SG, root, dont_merge=dont_merge)

    # Remove orphans
    SG.remove_nodes_from(list(nx.isolates(SG)))

    return SG


def tree_layout(G):

    def dfs_assignment(G, node, pos, next_x): 
        """
        Calculates the node's position and recursively calculates it's childrens positions.
        The y position is calculated from epsilon, while the x position is calculated by first
        assigning leaves integer positions, and the branches take the average of their children.
        """
        parent = list(G.predecessors(node))
        children = list(G.successors(node))

        # Calculate X positon
        if len(children) == 0: 
            x = next_x
            next_x += 1
        else: 
            # Get children to assign their X's, and take their mean
            for child in children: 
                pos, next_x = dfs_assignment(G, child, pos, next_x)
            x = np.mean([pos[child][0] for child in children])

        # Calculate Y position
        if len(parent) == 1:
            y = 1 / G[parent[0]][node]['weight']
        else:
            y = 9.758 

        pos[node] = (x, y)

        return pos, next_x

    root = [n for n, d in G.in_degree() if d==0][0]
    pos = {}
    pos, _ = dfs_assignment(G, root, pos, 0)

    return pos


def draw_circular_tree(G, original_G, df, column_names):
    dashboard_cluster_sizes = np.array([])

    def find_nodes_to_explain(node, max_ratio=9):
        children = list(G[node])
        if G.nodes[node]['size'] < 5000 or \
           len(children) != 2 or \
           G.nodes[children[0]]['size'] < 2000 or G.nodes[children[1]]['size'] < 2000 or \
           (G.nodes[children[0]]['size'] / G.nodes[children[1]]['size']) < 1 / max_ratio or \
           (G.nodes[children[0]]['size'] / G.nodes[children[1]]['size']) > max_ratio: 
               return False
        return True
        
    explain_nodes = filter(find_nodes_to_explain, list(G.nodes))

    fig = plt.figure(facecolor='white', figsize=(16, 8))
    ax  = plt.axes(frameon=False)

    pos = tree_layout(G)
    node_colors = [pos[n][1] for n in G.nodes]
    edge_widths = [G.nodes[n2]['size'] / 2000 for n1, n2 in G.edges]
    mynodes = nx.draw_networkx_nodes(G, pos, node_size=1500, edgelist=[], linewidths=1, edgecolors='k', node_color=node_colors, cmap=plt.cm.coolwarm, alpha=1.0, with_labels=False)
    nx.draw_networkx_edges(G, pos, edge_color='#B0B0B0', width=edge_widths, arrows=False, alpha=1.0, with_labels=False)
    cbar = plt.colorbar(mynodes)
    cbar.set_label("Epsilon parameter: the distance at which clusters merge or split", fontsize=16)

    # Draw node labels for nodes that are not hand-selected clusters
    node_labels = {k: f"{v}" for k, v in nx.get_node_attributes(G, 'size').items()}
    node_label_pos = {k: (v[0], v[1]) for k, v in pos.items()}
    nx.draw_networkx_labels(G, node_label_pos, labels=node_labels, font_color='black', fontsize=8)

    # Print explanations 
    for node in G:
        get_DT_for_node(df, original_G, node, column_names=column_names)

    plt.show()


def main():
    df, clusterer = dataset.default_dataset(paths=["../data/anonimized_io.csv"])
    print("Loaded dataset and HDBSCAN clusterer")

    ct = clusterer.condensed_tree_
    G  = ct.to_networkx()
    print("Converted condensed tree to networkX graph")

    split_multidegree_nodes(G)
    print("Split multi-degree nodes into multiple 2-degree nodes")

    CG = build_condensed_graph(G, 3., 1000, dont_merge=[])
    print("Built condensed graph with {} nodes and {} edges".format(len(CG.nodes), len(CG.edges)))

    print("Drawing the tree")
    log_columns = set([c for c in df.columns if 'perc' in c.lower()])
    draw_circular_tree(CG, G, df[log_columns], list(log_columns))


if __name__ == "__main__":
    main()

