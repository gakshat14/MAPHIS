import glob

citiesPath = r'C:\Users\hx21262\Dropbox\MAPHIS\Data\OS Maps\1880-1900\Scans'

citiesPaths = glob.glob(citiesPath+'/*')
nCities = len(citiesPaths)
nTiles = 0
nThumbnails = 335
weightThumbnail = 1.025
for cityPath in citiesPaths:

    nTiles += len(glob.glob(cityPath+'/*/*/*.tif'))

print(f'Budget : {nTiles} x {nThumbnails} x {weightThumbnail} (nTiles*nThumbnails*thumbnail Mb size) = {nTiles*nThumbnails*weightThumbnail:.2f} Mb = {nTiles*nThumbnails*weightThumbnail/1024:.2f} Gb = {nTiles*nThumbnails*weightThumbnail/(1024*1024):.2f} Tb')