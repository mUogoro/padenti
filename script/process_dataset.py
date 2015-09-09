import sys
from os import mkdir,listdir,sep
import numpy as np
from cv2 import imread,imwrite

# Download dataset at:
# http://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm#download
datasetPath = sys.argv[1]
outPath = sys.argv[2]
files = listdir(datasetPath)

try:
    mkdir(outPath)
except OSError: pass
try:
    mkdir(outPath+sep+"train")
except OSError: pass
try:
    mkdir(outPath+sep+"test")
except OSError: pass

testSplitSize = int(len(files)*0.1)
trainSplit = files[:-testSplitSize]
testSplit = files[len(files)-testSplitSize:]

def getDepthAndLabels(_file):
    depthAndLabels = imread(datasetPath+sep+_file, -1)
    bChannel = depthAndLabels[:,:,0]
    gChannel = depthAndLabels[:,:,1]
    rChannel = depthAndLabels[:,:,2]
    
    depth = bChannel.astype(np.uint16)+np.left_shift(gChannel.astype(np.uint16),8)
    depth[depth>2000] = 0

    handMask = rChannel>0
    bgMask = np.logical_and(np.logical_not(handMask),depth>0)

    labels = np.zeros((depth.shape[0],depth.shape[1],3), dtype=np.uint8)
    labels[handMask]=[0,0,255]
    labels[bgMask]=[255,0,0]

    return depth,labels

for _file in trainSplit:
    depth,labels = getDepthAndLabels(_file)
    imwrite(outPath+sep+"train"+sep+_file[6:-4]+"_labels.png", labels)
    imwrite(outPath+sep+"train"+sep+_file[6:-4]+"_depth.png", depth)

for _file in testSplit:
    depth,labels = getDepthAndLabels(_file)
    imwrite(outPath+sep+"test"+sep+_file[6:-4]+"_labels.png", labels)
    imwrite(outPath+sep+"test"+sep+_file[6:-4]+"_depth.png", depth)

exit(0)
