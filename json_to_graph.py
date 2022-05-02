import argparse
import pandas as pd
from pyprojroot import here
import pathlib
from datasets.datasetsFunctions import matchKeyToName
import constants.constants as constants
from graphs import graph
import json

classes = json.load(open(r'C:\Users\emile\MAPHIS\datasets\classifiedLayers\classes.json'))
inv_map = {v: k for k, v in classes.items()}

def construct_graph():
    tile_graph = graph.Graph()
    cityName = matchKeyToName(pathlib.Path(f'{args.datasetPath}/cityKey.json'), args.cityKey)
    dataframe = pd.read_json(here().joinpath(f'datasets/classifiedLayers/{cityName}/{args.tileName}.json')).transpose()
    for index, row in dataframe.iterrows():
        new_node = graph.Node(row['xTile'], row['yTile'],  row['class'])
        neighbours = getLabels(dataframe, row['xTile'], row['yTile'], constants.PROXIMITY)
        for index_n, row_n in neighbours.iterrows():
            x_ = row_n['xTile']
            y_ = row_n['yTile']
            label_ = row_n['class']
            neighbour_node = graph.Node(x_, y_, label_)
            new_node.add_neighbour(neighbour_node)
        tile_graph.add_node(new_node)

    tile_graph.display_graph()
    tile_graph.save_to_json('test.json')

def getLabels(dataframe:pd.DataFrame, x:int, y:int, distance:int) -> pd.DataFrame:
    allLabels = dataframe.query('xTile-@distance<@x<xTile+@distance and yTile-@distance<@y<yTile+@distance and xTile!=@x and yTile!=@y')
    meaningfullLabels = allLabels[allLabels['class'] != 0]
    return meaningfullLabels

def main(args):
    
    construct_graph()
    tile_graph = graph.Graph()
    tile_graph.load_from_json('test.json')
    
    tile_graph.display_graph()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetPath', required=False, type=str, default='datasets')
    parser.add_argument('--cityKey', required=False, type=str, default = '36')
    parser.add_argument('--tileName', required=False, type=str, default= '0105033010241')
    args = parser.parse_args()
    main(args)