from cProfile import label
import json
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from datasets.datasetsFunctions import matchKeyToName, openfile
classes = json.load(open(r'C:\Users\emile\MAPHIS\datasets\classifiedLayers\classes.json'))
inv_map = {v: k for k, v in classes.items()}

@dataclass
class Node():
    def __init__(self, x, y, label) -> None:
        self.x = x
        self.y = y
        self.label = label
        self.neighbours = []

    @property
    def to_string(self) -> str:
        return f'x : {self.x}, y : {self.y}, class : { inv_map[int(self.label)]}'
        
    @property
    def serialise_coords(self):
        node_dict = {
            'x':self.x,
            'y':self.y,
            'label':self.label
        }
        return node_dict

    @property
    def serialise_node(self) -> dict:
        return_dict = self.serialise_coords
        return_dict['neighbours'] = [n.serialise_coords for n in self.neighbours]
        return return_dict

    def add_neighbour(self, neighbour):
        self.neighbours.append(neighbour)

class Graph():
    def __init__(self) -> None:
        self.nodes = []
        self.edges = []
        self.sub_trees = []
        
    def add_node(self, node:Node) -> None:
        self.nodes.append(node)

    def __str__(self)-> str:
        for node in self.nodes:
            print(node.to_string) 
        return ''
    
    def connect(self, node):
        for sub_tree in self.sub_trees:
            if is_reachable(node, sub_tree):
                sub_tree.append(node)
                return True
        return False

    def make_sub_trees(self):
        for node in self.nodes:             
            if not self.connect(node):
                self.sub_trees.append([node])

    def save_to_json(self, savePath):
        return_list = [node.serialise_node for node in self.nodes]
        with open(f'{savePath}', 'w') as out_f:
            json.dump(return_list, out_f, indent = 4)

    def load_from_json(self, loadPath):
        load_list = json.load(open(loadPath))
        for node in load_list:
            new_node = Node(node['x'], node['y'], node['label'])
            for neighbour in node['neighbours']:
                new_node_n = Node(neighbour['x'], neighbour['y'], neighbour['label'])
                new_node.add_neighbour(new_node_n)
            self.add_node(new_node)

    '''def display_graph(self):
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        drawn = []
        for i, node in enumerate(self.nodes):
            plt.plot(int(node.x), int(node.y)   , marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
            plt.text(int(node.x)-20, int(node.y)-20, inv_map[node.label])
            for neighbour in node.neighbours:
                if not is_drawned([node.x, neighbour.x], [node.y, neighbour.y], drawn): 
                    plt.plot([node.x, neighbour.x], [node.y, neighbour.y], color='black')    
        
        plt.grid()
        plt.show()'''

    def display_graph(self):
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        drawn = []
        self.make_sub_trees()
        print(self.sub_trees)
        for sub_tree in self.sub_trees:
            c=np.random.rand(3,)
            for i, node in enumerate(sub_tree):                
                plt.plot(int(node.x), int(node.y)   , marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
                plt.text(int(node.x)-20, int(node.y)-20, inv_map[node.label])
                for neighbour in node.neighbours:
                    if not is_drawned([node.x, neighbour.x], [node.y, neighbour.y], drawn):
                        plt.plot([node.x, neighbour.x], [node.y, neighbour.y], c=c)    
        
        plt.grid()
        plt.show()

def is_drawned(arr_0:list, arr_1:list, drawn:list) -> bool:
    if [(arr_1[1], arr_0[0]), (arr_1[1], arr_0[0])]  in drawn:
        return True
    return False

def is_reachable(node:Node, sub_tree:list):
    for sub_node in sub_tree:
        if node in sub_node.neighbours:
            return True
    return False

def main():
    return 0

if __name__=='__main__':
    main()    