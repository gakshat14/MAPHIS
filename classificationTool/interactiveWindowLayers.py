import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
import glob
from PIL import Image
from matplotlib.figure import Figure
import shutil
import cv2
import numpy as np
from pathlib import Path
import json

tolerance = 10

class Application(ttk.Frame):
    def __init__(self, master:tk.Tk, cityName:str, datasetPath:Path, classifiedFolderPath:Path, tileFileFormat:str):
        super(Application, self).__init__()
        self.featureName = 'labels'
        self.savePathVisuals = Path(f'{datasetPath}/visuals/{self.featureName}')
        self.savePathVisuals.mkdir(parents=True, exist_ok=True)

        self.cityName = cityName
        self.datasetPath = datasetPath

        self.tileFileFormat = tileFileFormat
        self.cityPathMap = next(datasetPath.glob(f'cities/{cityName}/*/*'))
        self.allTilesPath = list(self.cityPathMap.glob(f'*{tileFileFormat}'))
        self.allTilesNames = [tilePath.stem for tilePath in self.allTilesPath]

        self.tilingParameters = json.load(open(datasetPath / 'layers/tilingParameters.json'))
        self.paddingX, self.paddingY = self.tilingParameters['paddingX'], self.tilingParameters['paddingY']

        self.cityPathLabel =  Path(f'{datasetPath}/layers/{self.featureName}/{cityName}')
        self.allLabelsToClassifyPath = list(self.cityPathLabel.glob(f'*.json'))
        self.allLabelsToClassifyNames =  [tilePath.stem for tilePath in self.allLabelsToClassifyPath]

        self.classifiedFolderPath = datasetPath / Path(f'classifiedLayers/{cityName}')
        self.classifiedFolderPath.mkdir(parents=True, exist_ok=True)

        self.setCurrentlyOpenedFile(self.allLabelsToClassifyNames[0])
        self.nLabels = len(self.currentlyOpenedLabels)
        self.loadClassifiedDict()

        self.currentCoordinates = self.getCurrentCoordinates()

        self.figThumbnail = Figure(figsize=(5, 4), dpi=100)
        self.canvaThumbnail = FigureCanvasTkAgg(self.figThumbnail, master)
        self.canvaThumbnail.get_tk_widget().grid(row=0,column=1)
        self.displayThumbnail()

        # Deal with buttons : put it on the grid (0,0) of the master by creating a Frame at that location
        rowButtonActions = 0
        rowIndexInfo = 1
        rowButtonPredefined0 = 2
        rowButtonPredefined1 = 3
        rowButtonPredefined2 = 4
        rowButtonPredefined3 = 5
        rowButtonPredefined4 = 6
        rowButtonPredefined5 = 7
        rowButtonPredefined6 = 8
        rowButtonPredefined7 = 9
        rowTileInfo = rowButtonPredefined7+ 1
        rowSave = rowTileInfo + 1

        self.buttonFrame = tk.Frame(master)
        self.buttonFrame.grid(row=rowButtonActions,column=0)

        self.textField = ttk.Entry(self.buttonFrame , text="Input feature name")
        self.textField.grid(row=rowButtonActions,column=0)

        self.saveButton = tk.Button(self.buttonFrame, text="Save and Update", command=lambda:[self.classify(self.textField.get()), self.clearTextInput(self.textField),self.updateCanvas()])
        self.saveButton.grid(row=rowButtonActions,column=1)

        self.fpButton = tk.Button(self.buttonFrame, text="False Positive", command=lambda:[self.classify("false positive"), self.updateCanvas()])
        self.fpButton.grid(row=rowButtonActions,column=2)

        self.currentIndexDisplay = tk.Label(self.buttonFrame, height = 1 , width = len(f'{self.nLabels}/ {self.nLabels}'), text=f'{self.currentThumbnailIndex} / {self.nLabels}')
        self.currentIndexDisplay.grid(row=rowIndexInfo, column=0)

        self.indexJumpTextBox = ttk.Entry(self.buttonFrame , text="Go to index")
        self.indexJumpTextBox.grid(row=rowIndexInfo,column=1)

        self.indexJumpButton = ttk.Button(self.buttonFrame , text="Jump to index", command=lambda:[self.updateCanvas(self.indexJumpTextBox.get()), self.clearTextInput(self.indexJumpTextBox)])
        self.indexJumpButton.grid(row=rowIndexInfo,column=2)

        self.indexJumpButton = ttk.Button(self.buttonFrame , text="Previous one", command=lambda:[self.updateCanvas(str(self.currentThumbnailIndex-1))])
        self.indexJumpButton.grid(row=rowIndexInfo,column=3)

        self.buttonLampPost = tk.Button(self.buttonFrame, text="L.P", command=lambda:[self.classify("lamppost"),self.updateCanvas()])
        self.buttonLampPost.grid(row=rowButtonPredefined0,column=0)
        self.buttonManHole = tk.Button(self.buttonFrame, text="M.H", command=lambda:[self.classify("manhole"),self.updateCanvas()])
        self.buttonManHole.grid(row=rowButtonPredefined0,column=1)
        self.buttonStone = tk.Button(self.buttonFrame, text="stone", command=lambda:[self.classify("stone"),self.updateCanvas()])
        self.buttonStone.grid(row=rowButtonPredefined0,column=2)
        self.buttonChimney = tk.Button(self.buttonFrame, text="chimney", command=lambda:[self.classify("chimney"), self.updateCanvas()])
        self.buttonChimney.grid(row=rowButtonPredefined0,column=3)
        self.buttonChy = tk.Button(self.buttonFrame, text="chy.", command=lambda:[self.classify("chy"), self.updateCanvas()])
        self.buttonChy.grid(row=rowButtonPredefined1,column=0)
        self.buttonHotel = tk.Button(self.buttonFrame, text="hotel", command=lambda:[self.classify("hotel"), self.updateCanvas()])
        self.buttonHotel.grid(row=rowButtonPredefined1,column=1)
        self.buttonChurch = tk.Button(self.buttonFrame, text="church", command=lambda:[self.classify("church"), self.updateCanvas()])
        self.buttonChurch.grid(row=rowButtonPredefined1,column=2)
        self.buttonWorkshop = tk.Button(self.buttonFrame, text="workshop", command=lambda:[self.classify("workshop"), self.updateCanvas()])
        self.buttonWorkshop.grid(row=rowButtonPredefined1,column=3)
        self.buttonFirepost = tk.Button(self.buttonFrame, text="firepost", command=lambda:[self.classify("firepost"), self.updateCanvas()])
        self.buttonFirepost.grid(row=rowButtonPredefined2,column=0)
        self.buttonRiver = tk.Button(self.buttonFrame, text="river", command=lambda:[self.classify("river"), self.updateCanvas()])
        self.buttonRiver.grid(row=rowButtonPredefined2,column=1)
        self.buttonSchool = tk.Button(self.buttonFrame, text="school", command=lambda:[self.classify("school"), self.updateCanvas()])
        self.buttonSchool.grid(row=rowButtonPredefined2,column=2)
        self.buttonBarrack = tk.Button(self.buttonFrame, text="barrack", command=lambda:[self.classify("barrack"), self.updateCanvas()])
        self.buttonBarrack.grid(row=rowButtonPredefined2,column=3)
        self.buttonWorkhouse = tk.Button(self.buttonFrame, text="workhouse", command=lambda:[self.classify("workhouse"), self.updateCanvas()])
        self.buttonWorkhouse.grid(row=rowButtonPredefined3,column=0)
        self.buttonMarket = tk.Button(self.buttonFrame, text="market", command=lambda:[self.classify("market"), self.updateCanvas()])
        self.buttonMarket.grid(row=rowButtonPredefined3,column=1)
        self.buttonChapel = tk.Button(self.buttonFrame, text="chapel", command=lambda:[self.classify("chapel"), self.updateCanvas()])
        self.buttonChapel.grid(row=rowButtonPredefined3,column=2)
        self.buttonBank = tk.Button(self.buttonFrame, text="bank", command=lambda:[self.classify("bank"), self.updateCanvas()])
        self.buttonBank.grid(row=rowButtonPredefined3,column=3)
        self.buttonPub = tk.Button(self.buttonFrame, text="pub", command=lambda:[self.classify("pub"), self.updateCanvas()])
        self.buttonPub.grid(row=rowButtonPredefined4,column=0)
        self.buttonPublicHouse = tk.Button(self.buttonFrame, text="P.H", command=lambda:[self.classify("public house"), self.updateCanvas()])
        self.buttonPublicHouse.grid(row=rowButtonPredefined4,column=1)
        self.buttonHotel = tk.Button(self.buttonFrame, text="hotel", command=lambda:[self.classify("hotel"), self.updateCanvas()])
        self.buttonHotel.grid(row=rowButtonPredefined4,column=2)
        self.buttonInn = tk.Button(self.buttonFrame, text="inn", command=lambda:[self.classify("inn"), self.updateCanvas()])
        self.buttonInn.grid(row=rowButtonPredefined4,column=3)
        self.buttonBath = tk.Button(self.buttonFrame, text="bath", command=lambda:[self.classify("bath"), self.updateCanvas()])
        self.buttonBath.grid(row=rowButtonPredefined5,column=0)
        self.buttonTheatre = tk.Button(self.buttonFrame, text="theatre", command=lambda:[self.classify("theatre"), self.updateCanvas()])
        self.buttonTheatre.grid(row=rowButtonPredefined5,column=1)
        self.buttonPolice = tk.Button(self.buttonFrame, text="police", command=lambda:[self.classify("police"), self.updateCanvas()])
        self.buttonPolice.grid(row=rowButtonPredefined5,column=2)
        self.buttonWharf = tk.Button(self.buttonFrame, text="wharf", command=lambda:[self.classify("wharf"), self.updateCanvas()])
        self.buttonWharf.grid(row=rowButtonPredefined5,column=3)
        self.buttonYard = tk.Button(self.buttonFrame, text="yard", command=lambda:[self.classify("yard"), self.updateCanvas()])
        self.buttonYard.grid(row=rowButtonPredefined6,column=0)
        self.buttonGreen = tk.Button(self.buttonFrame, text="green", command=lambda:[self.classify("green"), self.updateCanvas()])
        self.buttonGreen.grid(row=rowButtonPredefined6,column=1)
        self.buttonPark = tk.Button(self.buttonFrame, text="park", command=lambda:[self.classify("park"), self.updateCanvas()])
        self.buttonPark.grid(row=rowButtonPredefined6,column=2)
        self.buttonQuarry = tk.Button(self.buttonFrame, text="quarry", command=lambda:[self.classify("quarry"), self.updateCanvas()])
        self.buttonQuarry.grid(row=rowButtonPredefined6,column=3)
        self.buttonQuarry = tk.Button(self.buttonFrame, text="number", command=lambda:[self.classify("number"), self.updateCanvas()])
        self.buttonQuarry.grid(row=rowButtonPredefined7,column=0)

        self.tileNameDisplayed = tk.StringVar(self.buttonFrame)
        self.tileNameDisplayed.set(self.currentTileName) # default value

        self.dropdownMenu = tk.OptionMenu(self.buttonFrame, self.tileNameDisplayed, *self.allTilesNames)
        self.dropdownMenu.grid(row=rowTileInfo,column=0)

        self.changeTileButton = ttk.Button(self.buttonFrame , text="Change Tile", command=lambda:[self.changeTile(), self.updateIndex()])
        self.changeTileButton.grid(row=rowTileInfo,column=1)

        self.saveButton = ttk.Button(self.buttonFrame , text="Save progress", command=lambda:[self.saveProgress()])
        self.saveButton.grid(row=rowTileInfo,column=2)

        self.saveNameTxtField = ttk.Entry(self.buttonFrame , text="Input savename")
        self.saveNameTxtField.grid(row=rowSave,column=0)

        self.saveImageButton = ttk.Button(self.buttonFrame , text="Save Image", command=lambda:[self.saveImage(), self.clearTextInput(self.saveNameTxtField)])
        self.saveImageButton.grid(row=rowSave,column=1)


    def clearTextInput(self, textBoxAttribute):
        textBoxAttribute.delete(0, len(textBoxAttribute.get()))

    def saveImage(self):
        mat = self.currentlyOpenedMap[self.currentCoordinates['yTile']-self.paddingY:self.currentCoordinates['yTile']-self.paddingY+self.currentCoordinates['H'], self.currentCoordinates['xTile']-self.paddingX:self.currentCoordinates['xTile']-self.paddingX+self.currentCoordinates['W']]
        cv2.imwrite(f'{self.savePathVisuals}/{self.saveNameTxtField.get()}.jpg', mat)

    def fileOpenFunction(self, filePath:Path) -> np.float32:
        
        if filePath.suffix in ['.png', '.tif', '.jpg']:
            array = np.asarray(Image.open(filePath), np.uint8)[:,:,0]
            return array
        elif filePath.suffix == '.npz':
            return np.load(filePath, np.float32)['arr_0']
        else:
            print('Not Implemented')

    def updateIndex(self):
        self.currentIndexDisplay['text'] = f'({self.currentThumbnailIndex}) / ({self.nLabels})'

    def classify(self, savedClass:str):
        coordinates = self.currentCoordinates.copy()

        if savedClass.lower() not in self.featureList:        
            print(f"Adding class {savedClass} in {self.classifiedFolderPath.parent / Path(f'classes.json')}")
            self.featureList[savedClass] = len(self.featureList)
        
        coordinates['class'] = self.featureList[savedClass]
        self.classifiedDict[f'{self.currentThumbnailIndex}'] = coordinates

        try:
            self.progressDict['Not Classified'].remove(self.currentThumbnailIndex)
            self.progressDict['Classified'].append(self.currentThumbnailIndex)
        except ValueError:
            pass

    def updateCanvas(self, indexString=None):
        if indexString is not None:
            if indexString.isdigit() and int(indexString) < self.nLabels:
                    self.currentThumbnailIndex = int(indexString)
            else:
                print(f"Enter positive integers inferior to {self.nLabels} Only")
        else:
            if self.currentThumbnailIndex + 1< self.nLabels:
                self.currentThumbnailIndex+= 1
            else:
                print(f"Reached the end of the tile, not updating index")

        self.currentIndexDisplay['text'] = f'({self.currentThumbnailIndex}) / ({self.nLabels})'
        self.displayThumbnail()
    
    def changeTile(self):
        print(f"Loading Tile {self.tileNameDisplayed.get()}")
        try :
            self.setCurrentlyOpenedFile(self.tileNameDisplayed.get())
            self.loadClassifiedDict()
            self.displayThumbnail()        
        except IndexError:
            print(f'No {self.featureName} layer corresponding to tile {self.tileNameDisplayed.get()}')

    def saveProgress(self):
        print(self.classifiedFolderPath / f'{self.currentTileName}.json')
        writeJsonFile(Path(f'{self.classifiedFolderPath.parent}/classes.json'), {key: index for index, key in enumerate(self.featureList)})
        writeJsonFile(self.classifiedFolderPath / f'{self.currentTileName}.json', self.classifiedDict)
        writeJsonFile(self.classifiedFolderPath / f'{self.currentTileName}_progress.json', self.progressDict)
        print(f'Progress saved in {self.currentTileName}.json')

    def setCurrentlyOpenedFile(self, tileName:str):
        self.currentTileName = tileName
        try:
            self.currentIndex = self.allTilesNames.index(self.currentTileName)
        except ValueError:
            raise FileNotFoundError((f"{self.currentTileName}{self.tileFileFormat} can't be found, is it in {self.cityPathMap}?"))
        self.currentTilePath = self.allTilesPath[self.currentIndex]
        self.currentLabelFilePath = self.allLabelsToClassifyPath[self.currentIndex]
        self.currentlyOpenedMap = self.fileOpenFunction(self.currentTilePath)
        self.currentlyOpenedLabels = json.load(open(self.currentLabelFilePath))[self.featureName]
        self.featureList = openJsonFile(self.classifiedFolderPath.parent / Path(f'classes.json'))

    def getCurrentCoordinates(self)->dict:
        return self.currentlyOpenedLabels[f'{self.currentThumbnailIndex}']

    def displayThumbnail(self):
        self.currentCoordinates = self.getCurrentCoordinates()
        self.figThumbnail.clear()
        self.figThumbnail.add_subplot(111).imshow(self.currentlyOpenedMap[self.currentCoordinates['yTile']-self.paddingY:self.currentCoordinates['yTile']-self.paddingY+self.currentCoordinates['H'], self.currentCoordinates['xTile']-self.paddingX:self.currentCoordinates['xTile']-self.paddingX+self.currentCoordinates['W']])
        self.canvaThumbnail.draw()

    def loadClassifiedDict(self):
        #ADD FEATURENAME SPECIFICATION IN SAVEPATH
        print(f'Loading progress file from {self.classifiedFolderPath}')
        if Path(self.classifiedFolderPath / f'{self.currentTileName}.json').is_file()==False:
            print(f"No saved progression at in {self.classifiedFolderPath} folder, creating...")
            self.classifiedDict = {}
            writeJsonFile(self.classifiedFolderPath / f'{self.currentTileName}.json', self.classifiedDict)
        else:
            self.classifiedDict = json.load(open(self.classifiedFolderPath / f'{self.currentTileName}.json'))

        if Path(self.classifiedFolderPath / f'{self.currentTileName}_progress.json').is_file()==False:
            self.progressDict = {'Not Classified':[i for i in range(self.nLabels)],
                                'Classified': []}
            writeJsonFile(self.classifiedFolderPath / f'{self.currentTileName}.json', self.progressDict)
        else:
            self.progressDict = json.load(open(self.classifiedFolderPath / f'{self.currentTileName}_progress.json'))


        if len(self.progressDict['Not Classified']) !=0:
            print(f'Loading element number {self.currentThumbnailIndex}')
            self.currentThumbnailIndex = self.progressDict['Not Classified'][0]
        else:
            print(f'Reminder : all labels have been classified')
            self.currentThumbnailIndex = self.progressDict['Classified'][0]

def writeJsonFile(filePath, file):
    with open(filePath, 'w') as outfile:
        json.dump(file, outfile)

def openJsonFile(filePath):
    return json.load(open(filePath))

