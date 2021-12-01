# -*- coding: utf-8 -*-
import csv
from pathlib import Path, PurePath
import tkinter as tk
import argparse
import json
from pyprojroot import here

def matchKeyToName(pathToJsonfile:str, key : str):
    cityKeysFile = json.load(open(pathToJsonfile))
    return cityKeysFile[key]['Town']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifType', type=str, required=False, default='labels')
    parser.add_argument('--datasetPath', required=False, type=PurePath, default = here().joinpath('datasets'))
    parser.add_argument('--cityKey', type=str, required=False, default='36')
    parser.add_argument('--tileFileFormat', type=str, required=False, default='.jpg')
    args = parser.parse_args()
    
    cityName = matchKeyToName(f'{args.datasetPath}/cityKey.json', args.cityKey)

    datasetPath = Path(args.datasetPath)

    print(f'Classification Type : {args.classifType}')
    if args.classifType.lower() == 'labels':
        defaultFeatureList = ['false positive', 'manhole','lamppost', 'stone', 'chimney', 'chy', 'hotel', 
                            'church', 'workshop', 'firepost', 'river', 'school', 'barrack', 
                            'workhouse', 'market', 'chapel', 'bank', 'pub', 'public house', 
                            'inn', 'bath', 'theatre', 'police', 'wharf', 'yard', 'green', 'park', 'quarry', 'number']
        from interactiveWindowLabels import Application

    elif args.classifType.lower() == 'tiles':
        defaultFeatureList = ['rich residential neighborhood', 'poor residential neighborhood', 'industrial district',
                               'peri-urban district',  'farm and forest']
        from interactiveWindowTiles import Application

    elif args.classifType.lower() == 'contours':
        defaultFeatureList = ['interesting','not interesting', 'tree', 'factory', 'villa']
        from interactiveWindowContours import Application
        
    else:
        raise ValueError ("Has to be contours, tiles or labels")

    classifiedFolderPath = datasetPath / f'classified{args.classifType.capitalize()}'
    if not Path(classifiedFolderPath / f'classes.json').is_file():
        parsedList = {key:i for i, key in enumerate(defaultFeatureList)}
        with open(Path(classifiedFolderPath / f'classes.json'), 'w') as outfile:
            json.dump(parsedList, outfile)

    root = tk.Tk()
    app = Application(root, cityName, datasetPath, classifiedFolderPath, args.tileFileFormat)
    root.mainloop()

if __name__=='__main__':
    main()