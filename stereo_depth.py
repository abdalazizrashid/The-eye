import sys
import numpy as np
import cv2


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

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# TODO: Use more stable identifiers
left = cv2.VideoCapture(4)
right = cv2.VideoCapture(6)

# Increase the resolution
left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# The distortion in the left and right edges prevents a good calibration, so
# discard the edges
CROP_WIDTH = 960
def cropHorizontal(image):
    return image[:,
            int((CAMERA_WIDTH-CROP_WIDTH)/2):
            int(CROP_WIDTH+(CAMERA_WIDTH-CROP_WIDTH)/2)]

# TODO: Why these values in particular?
# TODO: Try applying brightness/contrast/gamma adjustments to the images
#stereoMatcher = cv2.StereoBM_create()
#stereoMatcher.setMinDisparity(8)
#stereoMatcher.setNumDisparities(128)  #128
#stereoMatcher.setBlockSize(27)  #21
#stereoMatcher.setROI1(leftROI)
#stereoMatcher.setROI2(rightROI)
#stereoMatcher.setSpeckleRange(16)  #16
#stereoMatcher.setSpeckleWindowSize(45)  #45


# Create StereoSGBM and prepare all parameters
window_size = 5
min_disp = 2
num_disp = 130-min_disp
stereoMatcher = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 7,
    P1 = 8*3*window_size**2,
P2 = 32*3*window_size**2)

# Grab both frames first, then retrieve to minimize latency between cameras
while(True):
    if not left.grab() or not right.grab():
        print("No more frames")
        break

    _, leftFrame = left.retrieve()
    leftFrame = cropHorizontal(leftFrame)
    leftHeight, leftWidth = leftFrame.shape[:2]
    _, rightFrame = right.retrieve()
    rightFrame = cropHorizontal(rightFrame)
    rightHeight, rightWidth = rightFrame.shape[:2]

    if (leftWidth, leftHeight) != imageSize:
        print("Left camera has different size than the calibration data")
        break

    if (rightWidth, rightHeight) != imageSize:
        print("Right camera has different size than the calibration data")
        break

    fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
    fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)
    cv2.imwrite('left.jpg', fixedLeft)
    cv2.imwrite('right.jpg', fixedRight)

    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    depth = stereoMatcher.compute(grayLeft, grayRight)
    arr = np.uint8(depth)
    imC = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
    #cv2.imshow('left', fixedLeft)
    #cv2.moveWindow('left', 40, 30)  # Move it to (40,30)
    #cv2.imshow('right', fixedRight)
    #cv2.moveWindow('right', 40, 30)  # Move it to (40,30)
    #cv2.imshow('depth', depth / DEPTH_VISUALIZATION_SCALE)
    #cv2.moveWindow('colored', 100, 500)  # Move it to (40,30)
    #cv2.moveWindow('colored', 100, 500)  # Move it to (40,30)

    #cv2.imshow('colored', imC)
##############################################################################################################################
    # Filtering
    #matcher_left = depth
    kernel = np.ones((3, 3), np.uint8)
    stereoR = cv2.ximgproc.createRightMatcher(stereoMatcher)  # Create another stereo for right this time
    wls = cv2.ximgproc.createDisparityWLSFilter(stereoMatcher)

    lmbda = 80000
    sigma = 1.8
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereoMatcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # Compute the 2 images for the Depth_image
    dispL = depth
    dispR = stereoR.compute(grayRight, grayLeft)
    dispL = np.int16(dispL)
    dispR = np.int16(dispR)

    # Using the WLS filter
    filteredImg = wls_filter.filter(dispL, grayLeft, None, dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    cv2.imshow('Disparity Map', filteredImg)
    disp = ((depth.astype(np.float32) / 16) - 4) / 64  # Calculation allowing us to have 0 for the most distant object able to detect
    # Resize the image for faster executions
    dispR = cv2.resize(disp, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)

    filteredDisp = wls.filter(grayLeft, leftFrame, disparity_map_right=grayRight)

    closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE,
                               kernel)  # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)

    dispc = (closing - closing.min()) * 255
    dispC = dispc.astype(
        np.uint8)  # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
    disp_Color = cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)  # Change the Color of the Picture into an Ocean Color_Map
    filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_JET)

    cv2.imshow('Filtered Color Depth', filt_Color)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left.release()
right.release()
cv2.destroyAllWindows()