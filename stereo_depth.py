import sys
import numpy as np
import cv2

name = '1'
REMAP_INTERPOLATION = cv2.INTER_LINEAR

DEPTH_VISUALIZATION_SCALE = 2048

if len(sys.argv) != 2:
    print("Syntax: {0} CALIBRATION_FILE".format(sys.argv[0]))
    sys.exit(1)

calibration = np.load(sys.argv[1], allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])
#print(imageSize, leftMapX, leftMapY, leftROI, rightMapX, rightMapY, rightROI)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# TODO: Use more stable identifiers
#left = cv2.VideoCapture(6)
#right = cv2.VideoCapture(0)
path_l = 'remap/captures/left/000088.jpg'
path_r = 'remap/captures/right/000088.jpg'

left = cv2.imread(path_l)
right = cv2.imread(path_r)


# The distortion in the left and right edges prevents a good calibration, so
# discard the edges
CROP_WIDTH = 960
def cropHorizontal(image):
    return image[:,
            int((CAMERA_WIDTH-CROP_WIDTH)/2):
            int(CROP_WIDTH+(CAMERA_WIDTH-CROP_WIDTH)/2)]





# Grab both frames first, then retrieve to minimize latency between cameras
leftFrame = left
leftFrame = cropHorizontal(leftFrame)
leftHeight, leftWidth = leftFrame.shape[:2]
rightFrame = right
rightFrame = cropHorizontal(rightFrame)
rightHeight, rightWidth = rightFrame.shape[:2]
if (leftWidth, leftHeight) != imageSize:
    print("Left camera has different size than the calibration data")
if (rightWidth, rightHeight) != imageSize:
    print("Right camera has different size than the calibration data")
fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)
cv2.imwrite('./remap/remapped/left/' + str(name) + '.jpg', fixedLeft)
cv2.imwrite('./remap/remapped/right/' + str(name) + '.jpg', fixedRight)



