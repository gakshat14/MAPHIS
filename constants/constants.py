# All constants shared accross models are defined here

# Features Names
FEATURENAMES = ( 'trees', 'buildings', 'labels' )

# Area threshold (surfaces)
AREASHAPES = {'labels':0, 'trees':0, 'buildings':100}

# Colors to draw each features
COLORDICT = {'labels':(255,0,0), 'trees':(0,255,0), 'buildings':(0,0,255)}

# Tile sizes
TILEHEIGHT = 7590

TILEWIDTH = 11400

# Thumbnail parameters
KERNELSIZE = 512
WIDTHPADDING=100
HEIGHTPADDING = 157
WIDTHSTRIDE = 50
HEIGHTSTRIDE = 50
NCOLS = 24
NROWS = 16

PROXIMITY = 250