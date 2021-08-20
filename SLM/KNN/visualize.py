# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/8 20:45

from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx


def draw(node):
    r"""
    To visualize binary tree.

    Parameters
    ----------
    node: BTNode
        Binary tree root node.
    """
    saw = defaultdict(int)

    def create_graph(G, bt_node, p_name, pos_dict, x=0, y=0, layer=1):
        if bt_node is None:
            return
        name = str(bt_node.key)
        saw[name] += 1
        if name in saw.keys():
            name += ' ' * saw[name]

        G.add_edge(p_name, name)
        pos_dict[name] = (x, y)

        l_x, l_y = x - 2 / 3 ** layer, y - 1
        l_layer = layer + 1
        create_graph(G, bt_node.lchild, name, x=l_x, y=l_y, pos_dict=pos_dict, layer=l_layer)

        r_x, r_y = x + 2 / 3 ** layer, y - 1
        r_layer = layer + 1
        create_graph(G, bt_node.rchild, name, x=r_x, y=r_y, pos_dict=pos_dict, layer=r_layer)
        return G, pos_dict

    graph = nx.DiGraph()
    graph, pos = create_graph(graph, node, "     ", {})
    pos["     "] = (0, 0)
    fig, ax = plt.subplots(figsize=(8, 8))  # 比例可以根据树的深度适当调节
    nx.draw_networkx(graph, pos, ax=ax, node_size=1100, node_color='#ff0099', alpha=0.7)
    plt.show()
