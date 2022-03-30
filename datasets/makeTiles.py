import argparse
import json

def main():
    paddingX, paddingY = 100,157
    kernelSize = 512
    strideX = 50
    strideY = 50
    nCols = int((2*paddingX+11400-kernelSize)/(kernelSize-strideX))
    nRows = int((2*paddingY+7590-kernelSize)/(kernelSize-strideY))
    tilingDict = {'height':2*paddingY+7590 , 'width':2*paddingX+11400, 'kernelSize':kernelSize, 'paddingX':paddingX, 'paddingY':paddingY,'strideX':strideX, 'strideY':strideY, 'nCols':nCols, 'nRows':nRows}
    tilingDict['coordinates'] = {}
    nTiles = 0
    for rowIndex in range(nRows+1):
            yLow  = (kernelSize - strideY)*rowIndex
            yHigh = kernelSize*(rowIndex+1) - strideY*rowIndex
            for colIndex in range(nCols+1):
                xLow  = (kernelSize - strideX)*colIndex
                xHigh = kernelSize*(colIndex+1) - strideX*colIndex
                tilingDict['coordinates'][nTiles] = {'yLow':yLow, 'yHigh':yHigh, 'xLow':xLow, 'xHigh':xHigh}
                nTiles+=1
    with open(f'tilingParameters.json', 'w') as outfile:
        json.dump(tilingDict, outfile)

if __name__=='__main__':
    main()


