# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/8 20:15

import math

from SLM.KNN.visualize import draw

r"""
<<Statistics Learning Method>> by Li Hang.
Chapter 3: k-nearest neighbor (k-NN).

kd-tree to implement K-NN.
"""


class BTNode(object):
    def __init__(self, key, dim):
        r"""
        Definition of kd-tree node.

        :param key: tuple
            A tuple, indicating a vector

        :param dim: int
            Choose the dim-th dimension as the axis of segmentation
            when creating kd-tree.
        """
        self.key = key
        self.lchild = None
        self.rchild = None
        self.dim = dim

    def __str__(self):
        return 'BTNode' + str(self.key)


def kd_tree_create(dataset: list, depth: int, k: int):
    r"""
    Create kd-tree by dataset.

    :param dataset: list
        A list of data set, whose every element is a tuple indicating
        a vector

    :param depth: int
        Current tree node depth. Note: Depth of the root node is 0.

    :param k: int
        Dimension of the element in dataset.

    :return: BTNode
        The root node of kd-tree.
    """
    length = len(dataset)
    if length == 0:
        return None
    dim = depth % k
    dataset.sort(key=lambda x: x[dim])
    root_key = dataset[length // 2]
    root = BTNode(root_key, dim)
    root.lchild = kd_tree_create(dataset[:length // 2], depth + 1, k)
    root.rchild = kd_tree_create(dataset[length // 2 + 1:], depth + 1, k)
    return root


def search_leaf_node(kd_tree_node: BTNode, target, path):
    r"""
    Search the nearest neighbor to the target point in all of
    the kd-tree's leaf nodes.

    :param kd_tree_node: BTNode
        A child tree with kd_tree_node as root.

    :param target: tuple
        A target point, indication a vector

    :param path: list -> BTNode
        The searching path from kd_tree_node to the nearest leaf node.

    :return: (tuple, list -> BTNode)
        Key of the nearest leaf node.
        The searching path.
    """
    dim = kd_tree_node.dim
    path.append(kd_tree_node)
    if target[dim] < kd_tree_node.key[dim]:
        if kd_tree_node.lchild:
            return search_leaf_node(kd_tree_node.lchild, target, path)
        else:
            return kd_tree_node.key, path
    else:
        if kd_tree_node.rchild:
            return search_leaf_node(kd_tree_node.rchild, target, path)
        else:
            return kd_tree_node.key, path


def distance(target, source):
    r"""
    Calculate the distance between target point and source point.

    :param target: tuple
        Target point.

    :param source: tuple
        Source point.

    :return: float
        return the distance between target point and source point.
    """
    return math.sqrt(sum([(x - y) ** 2 for x, y in zip(target, source)]))


def search_nn(path, target, visited: list, min_distance=float('inf'), nn=None):
    r"""
    Searching the nearest neighbor to the target point in all of
    kd-tree nodes, not only leaf nodes.

    :param path: list -> BTNode
        Searching path.
        See search_leaf_node(kd_tree_node: BTNode, target, path).

    :param target: tuple
        Target point.

    :param visited: list -> BTNode
        A list to justify whether a kd-tree node has been visited.
        The node has been visited if condition(node in visited) is True.

    :param nn: BTNode
        The current nearest neighbor to the target point.

    :param min_distance: float
        Distance between nn and the target point.

    :return: (tuple, float)
        Key of the nearest neighbor to the target point in all
        of kd-tree nodes.
        Distance between the nearest neighbor and the target point.
        
    """
    leaf_node = path[-1]
    visited.append(leaf_node)
    if min_distance > (temp := distance(leaf_node.key, target)):
        nn = leaf_node
        min_distance = temp

    if len(path) == 1:
        return nn.key, min_distance

    parent = path[-2]
    another = parent.lchild if nn == parent.rchild else parent.rchild
    if min_distance > (temp := distance(parent.key, target)):
        nn = parent
        min_distance = temp

    if not another or another in visited or abs(parent.key[parent.dim] - target[parent.dim]) >= min_distance:
        return search_nn(path[:-1], target, visited, min_distance, nn)
    else:
        sub_key, temp_path = search_leaf_node(another, target, [])
        return search_nn(path[:-1] + temp_path, target, visited, min_distance, nn)


def main(draw_show=True):
    dataset = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    kd_tree = kd_tree_create(dataset, 0, 2)
    if draw_show:
        draw(kd_tree)

    target = (12, 10)
    print('target:', target)
    leaf, path = search_leaf_node(kd_tree, target, [])
    print('Path:', end=' ')
    for p in path:
        print(p, end=' ')
    print()

    nn, min_distance = search_nn(path, target, [])
    print('nn:', nn)
    print('min_distance:', min_distance)


if __name__ == '__main__':
    main(draw_show=True)
