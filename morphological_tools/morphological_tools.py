import cv2
import numpy as np

def dilation(src:np.float32, dilateSize=1):
    """Dilatation operation, can be applied to the image for map "cleaning"

    Args:
        src (np.float32): source image
        dilateSize (int, optional): Dilatation kernel size. Defaults to 1.

    Returns:
        _type_: dilated image
    """
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilateSize + 1, 2 * dilateSize + 1),
                                    (dilateSize, dilateSize))
    return cv2.dilate(src.astype('uint8'), element)

def erosion(src, dilateSize=1):
    """Erosion operation, can be applied to the image for map "cleaning"

    Args:
        src (np.float32): source image
        dilateSize (int, optional): Erosion kernel size. Defaults to 1.

    Returns:
        _type_: eroded image
    """
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilateSize + 1, 2 * dilateSize + 1),
                                    (dilateSize, dilateSize))
    return cv2.erode(src.astype('uint8'), element)