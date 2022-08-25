    ybox.append(bucket)
    print(ybox)
    # ypoint.append(ybox[1::2])
xbox = []
ybox = []
xpoint = []
ypoint = []
for bucket in pd:
    xbox.append(bucket)
    xpoint.append(xbox[::2])
    print(xpoint)
print(xpoint)
for bucket in pd:
    xbox.append(bucket[::2])
    print(xbox)
print(xbox)
for bucket in range(len(pd)):
    xbox.append(pd[::2])
    print(xbox)
print(pd)
xbox = []
ybox = []
xpoint = []
ypoint = []
for bucket in range(len(pd)):
    xbox.append(pd[::2])
    print(xbox)
xbox = []
ybox = []
xpoint = []
ypoint = []
for bucket in range(len(pd)):
    xbox.append(bu1[::2])
    print(xbox)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
xbox = []
ybox = []
xpoint = []
ypoint = []
for bucket in range(len(pd)):
    xpoint.append(pd[::2])
    print(xpoint)
xbox = []
ybox = []
xpoint = []
ypoint = []
for bucket in pd:
    xpoint.append(pd[::2])
    print(xpoint)
xbox = []
ybox = []
xpoint = []
ypoint = []
for bucket in pd:
    xpoint.append(bucket[::2])
    print(xpoint)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
xpoint = []
ypoint = []
for bucket in pd:
    xpoint.append(bucket[::2])
    ypoint.append(bucket[1::2])
    
    x1 = xMax(xpoint)
    y1 = yMax(ypoint)
    x2 = xMin(xpoint)
    y2 = yMin(ypoint)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
import cv2
import numpy as np
import json

with open("/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.json", "r") as f:
    json_data = json.load(f)

load_json_data = json_data
polygon_data = load_json_data["annotations"]

pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))

print(pd)
#bbox = []
#for i in range(len(pd)):
#    val = pd -> bbox
#    bbox.append(val)
#    val로 rectangle


#xpoint = bu1["polygon"][::2]
#ypoint = bu1["polygon"][1::2]




def xMax(xpoint):
    xmax = xpoint[0]
    for i in xpoint:
        if xmax < i:
            xmax = i
    return xmax

def yMax(ypoint):
    ymax = ypoint[0]
    for i in ypoint:
        if ymax < i:
            ymax = i
    return ymax

def xMin(xpoint):
    xmin = xpoint[0]
    for i in xpoint:
        if xmin > i:
            xmin = i
    return xmin

def yMin(ypoint):
    ymin = ypoint[0]
    for i in ypoint:
        if ymin > i:
            ymin = i
    return ymin


# cv2
img_path = "/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.jpg"
clicked_points = []
ird = cv2.imread(img_path)
reset = ird.copy();
points = []

xpoint = []
ypoint = []
for bucket in pd:
    xpoint.append(bucket[::2])
    ypoint.append(bucket[1::2])
    
    x1 = xMax(xpoint)
    y1 = yMax(ypoint)
    x2 = xMin(xpoint)
    y2 = yMin(ypoint)
    
    img = cv2.rectangle(ird, (x1[0], y1[0]), (x2[0], y2[0]), (255, 0, 0), 7)
xpoint = []
ypoint = []
for bucket in pd:
    xpoint.append(bucket[::2])
    ypoint.append(bucket[1::2])
    
    x1 = xMax(xpoint)
    y1 = yMax(ypoint)
    x2 = xMin(xpoint)
    y2 = yMin(ypoint)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
import cv2
import numpy as np
import json

with open("/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.json", "r") as f:
    json_data = json.load(f)

load_json_data = json_data
polygon_data = load_json_data["annotations"]

pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))

print(pd)
#bbox = []
#for i in range(len(pd)):
#    val = pd -> bbox
#    bbox.append(val)
#    val로 rectangle


#xpoint = bu1["polygon"][::2]
#ypoint = bu1["polygon"][1::2]




def xMax(xpoint):
    xmax = xpoint[0]
    for i in xpoint:
        if xmax < i:
            xmax = i
    return xmax

def yMax(ypoint):
    ymax = ypoint[0]
    for i in ypoint:
        if ymax < i:
            ymax = i
    return ymax

def xMin(xpoint):
    xmin = xpoint[0]
    for i in xpoint:
        if xmin > i:
            xmin = i
    return xmin

def yMin(ypoint):
    ymin = ypoint[0]
    for i in ypoint:
        if ymin > i:
            ymin = i
    return ymin


# cv2
img_path = "/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.jpg"
clicked_points = []
ird = cv2.imread(img_path)
reset = ird.copy();
points = []

xpoint = []
ypoint = []
for bucket in pd:
    xpoint.append(bucket[::2])
    ypoint.append(bucket[1::2])
    
    x1 = xMax(xpoint)
    y1 = yMax(ypoint)
    x2 = xMin(xpoint)
    y2 = yMin(ypoint)
    
    img = cv2.rectangle(ird, (x1[0], y1[0]), (x2[0], y2[0]), (255, 0, 0), 7)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
import cv2
import numpy as np
import json

with open("/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.json", "r") as f:
    json_data = json.load(f)

load_json_data = json_data
polygon_data = load_json_data["annotations"]

pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))

print(pd)
#bbox = []
#for i in range(len(pd)):
#    val = pd -> bbox
#    bbox.append(val)
#    val로 rectangle


#xpoint = bu1["polygon"][::2]
#ypoint = bu1["polygon"][1::2]




def xMax(xpoint):
    xmax = xpoint[0]
    for i in xpoint:
        if xmax < i:
            xmax = i
    return xmax

def yMax(ypoint):
    ymax = ypoint[0]
    for i in ypoint:
        if ymax < i:
            ymax = i
    return ymax

def xMin(xpoint):
    xmin = xpoint[0]
    for i in xpoint:
        if xmin > i:
            xmin = i
    return xmin

def yMin(ypoint):
    ymin = ypoint[0]
    for i in ypoint:
        if ymin > i:
            ymin = i
    return ymin


# cv2
img_path = "/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.jpg"
clicked_points = []
ird = cv2.imread(img_path)
reset = ird.copy();
points = []


xpoint = []
ypoint = []
for bucket in pd:
    xpoint.append(bucket[::2])
    ypoint.append(bucket[1::2])
    
    x1 = xMax(xpoint)
    y1 = yMax(ypoint)
    x2 = xMin(xpoint)
    y2 = yMin(ypoint)
    
    img = cv2.rectangle(ird, (x2[0], y2[0]), (x1[0], y1[0]), (255, 0, 0), 7)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
xbox = []
ybox = []
xpoint = []
ypoint = []
for bucket in pd:
    xbox.append(bucket[::2])
    ybox.append(bucket[1::2])
print(xbox[1])
print(xbox[0])
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
f = open("/Users/daewon/desktop/test.txt", "w", encoding = "UTF8")
f.write("xMin yMin xMax yMax")
f.close()
sizef.append(x1)
sizef = []
for i in range(len(pd)):
    xpoint = xbox[i]
    ypoint = ybox[i]
    
    x1 = xMax(xpoint)
    y1 = yMax(ypoint)
    x2 = xMin(xpoint)
    y2 = yMin(ypoint)
    sizef.append(x1)
import cv2
import numpy as np
import json

with open("/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.json", "r") as f:
    json_data = json.load(f)

load_json_data = json_data
polygon_data = load_json_data["annotations"]

pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))


def xMax(xpoint):
    xmax = xpoint[0]
    for i in xpoint:
        if xmax < i:
            xmax = i
    return xmax

def yMax(ypoint):
    ymax = ypoint[0]
    for i in ypoint:
        if ymax < i:
            ymax = i
    return ymax

def xMin(xpoint):
    xmin = xpoint[0]
    for i in xpoint:
        if xmin > i:
            xmin = i
    return xmin

def yMin(ypoint):
    ymin = ypoint[0]
    for i in ypoint:
        if ymin > i:
            ymin = i
    return ymin


# cv2
img_path = "/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.jpg"
clicked_points = []
ird = cv2.imread(img_path)
reset = ird.copy();
points = []

xbox = []
ybox = []
xpoint = []
ypoint = []
for bucket in pd:
    xbox.append(bucket[::2])
    ybox.append(bucket[1::2])

sizef = []
for i in range(len(pd)):
    xpoint = xbox[i]
    ypoint = ybox[i]
    
    x1 = xMax(xpoint)
    y1 = yMax(ypoint)
    x2 = xMin(xpoint)
    y2 = yMin(ypoint)
    sizef.append(x1)
    
    
    img = cv2.rectangle(ird, (x2, y2), (x1, y1), (255, 0, 0), 7)

data = "xMin yMin xMax yMax"
f = open("/Users/daewon/desktop/test.txt", "w", encoding = "UTF8")
f.write(data)

f.close()

cv2.imshow("image", ird)
cv2.namedWindow("image")
# cv2.setMouseCallback("image", MouseLeftClick)

cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np
import json

with open("/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.json", "r") as f:
    json_data = json.load(f)

load_json_data = json_data
polygon_data = load_json_data["annotations"]

pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))


def xMax(xpoint):
    xmax = xpoint[0]
    for i in xpoint:
        if xmax < i:
            xmax = i
    return xmax

def yMax(ypoint):
    ymax = ypoint[0]
    for i in ypoint:
        if ymax < i:
            ymax = i
    return ymax

def xMin(xpoint):
    xmin = xpoint[0]
    for i in xpoint:
        if xmin > i:
            xmin = i
    return xmin

def yMin(ypoint):
    ymin = ypoint[0]
    for i in ypoint:
        if ymin > i:
            ymin = i
    return ymin


# cv2
img_path = "/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.jpg"
clicked_points = []
ird = cv2.imread(img_path)
reset = ird.copy();
points = []

xbox = []
ybox = []
xpoint = []
ypoint = []
for bucket in pd:
    xbox.append(bucket[::2])
    ybox.append(bucket[1::2])

sizef = []
for i in range(len(pd)):
    xpoint = xbox[i]
    ypoint = ybox[i]
    
    x1 = xMax(xpoint)
    y1 = yMax(ypoint)
    x2 = xMin(xpoint)
    y2 = yMin(ypoint)
    sizef.append(x1)
    
    
    img = cv2.rectangle(ird, (x2, y2), (x1, y1), (255, 0, 0), 7)
sizef.append(x1)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
print(polygon_data)
findStr = "annotations" in polygon_data

print(findStr)
import cv2
import numpy as np
import json
import os.path

with open("/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_152433.json", "r") as f:
    json_data = json.load(f)


load_json_data = json_data
polygon_data = load_json_data["annotations"]
findStr = "annotations" in polygon_data

print(findStr)

pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))


def xMax(xpoint):
    xmax = xpoint[0]
    for i in xpoint:
        if xmax < i:
            xmax = i
    return xmax

def yMax(ypoint):
    ymax = ypoint[0]
    for i in ypoint:
        if ymax < i:
            ymax = i
    return ymax

def xMin(xpoint):
    xmin = xpoint[0]
    for i in xpoint:
        if xmin > i:
            xmin = i
    return xmin

def yMin(ypoint):
    ymin = ypoint[0]
    for i in ypoint:
        if ymin > i:
            ymin = i
    return ymin
load_json_data = json_data
polygon_data = load_json_data["annotations"]
findStr = "annotations" in load_json_data

print(findStr)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')

## ---(Fri Aug 19 14:31:15 2022)---
runcell(0, '/Users/daewon/.spyder-py3/OpenCV/object_detection.py')

## ---(Fri Aug 19 16:31:49 2022)---
import cv2
import numpy as np
import json
import os.path

filePath = "/Users/daewon/desktop/deep-i/python/PVC파이프/IMG_9046.json"
img_path = "/Users/daewon/desktop/deep-i/python/PVC파이프/IMG_9046.jpg"

with open(filePath, "r") as f:
    json_data = json.load(f)


load_json_data = json_data
polygon_data = load_json_data["annotations"]
findStr = "annotations" in load_json_data

classNum = 0;

if findStr:
    classNum = 0;

pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))

pdnp = np.array(pd)
print(pdnp)

## ---(Mon Aug 22 11:24:10 2022)---
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
import cv2
import numpy as np
import json

with open("/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.json", "r") as f:
    json_data = json.load(f)

load_json_data = json_data
polygon_data = load_json_data["annotations"]

pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))

npd = np.array(pd[::2], pd[1::2])
npd = np.array(pd[::2])
npd = np.array(pd[::2], pd[1::2])
xpd = npd[::2]
npd = np.array(pd[::2], pd[1::2])
npd = np.array(pd)
npd = np.array(pd)
xpd = npd[::2, ::2]
ypd = npd[1::2, 1::2]
xpd = npd[::2, ::2]
print(npd.shape)
xpd = npd[::2]
npd = np.array(pd[0], pd[1])
print(pd[0])
print(pd[1])
npd = np.array([pd[0], pd[1]])
import cv2
import numpy as np
import json

with open("/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.json", "r") as f:
    json_data = json.load(f)

load_json_data = json_data
polygon_data = load_json_data["annotations"]

pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))

npd = np.array([pd[0], pd[1]])
print(npd.shape)
print(npd.shape)
npd = np.array([pd[0], pd[1]])
npd = np.array([pd[0], pd[1]])
print(npd)
xpd = npd[::2]
print(npd[0])
print(npd[1])
print(np.shape(npd))
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(xMax(xpd))
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
import cv2
import numpy as np
import json

with open("/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.json", "r") as f:
    json_data = json.load(f)

load_json_data = json_data
polygon_data = load_json_data["annotations"]

pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))

npd = np.array([pd[0], pd[1]])
print(npd[0])
print(npd[1])
print(np.shape(npd))
xpd = npd[0][::2]


def xMax(xpoint):
    xmax = xpoint[0]
    for i in xpoint:
        if xmax < i:
            xmax = i
    return xmax

nM = xMax(xpd)
print(nM)
print(xpd[0])
ypd = npd[0][1::2]
print(xpd[0])
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
textPath = "/Users/daewon/desktop/test.txt"
data = "class xMin yMin xMax yMax"
if os.path.isfile(textPath):
    f = open(textPath, "a", encoding = "UTF8")
else:
    f = open(textPath, "w", encoding = "UTF8")
    f.write(data)
import numpy as np
import json
import os.path

with open("/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.json", "r") as f:
    json_data = json.load(f)

load_json_data = json_data
polygon_data = load_json_data["annotations"]

pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))



def xMax(xpoint):
    xmax = xpoint[0]
    for i in xpoint:
        if xmax < i:
            xmax = i
    return xmax


def yMax(ypoint):
    ymax = ypoint[0]
    for i in ypoint:
        if ymax < i:
            ymax = i
    return ymax

def xMin(xpoint):
    xmin = xpoint[0]
    for i in xpoint:
        if xmin > i:
            xmin = i
    return xmin

def yMin(ypoint):
    ymin = ypoint[0]
    for i in ypoint:
        if ymin > i:
            ymin = i
    return ymin


# cv2
img_path = "/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.jpg"
clicked_points = []
ird = cv2.imread(img_path)
reset = ird.copy();
points = []

xbox = []
ybox = []
xpoint = []
ypoint = []
for bucket in pd:
    xbox.append(bucket[::2])
    ybox.append(bucket[1::2])

findStr = "annotations" in load_json_data
classNum = 0;
if findStr:
    classNum = 0;


npd = np.array([pd[0], pd[1]], dtype=object)
textPath = "/Users/daewon/desktop/test.txt"
data = "class xMin yMin xMax yMax"
if os.path.isfile(textPath):
    f = open(textPath, "a", encoding = "UTF8")
else:
    f = open(textPath, "w", encoding = "UTF8")
    f.write(data)
textPath = "/Users/daewon/desktop/test.txt"
data = "class xMin yMin xMax yMax"
if os.path.isfile(textPath):
    f = open(textPath, "a", encoding = "UTF8")
else:
    f = open(textPath, "w", encoding = "UTF8")
    f.write(data)
runcell(0, '/Users/daewon/.spyder-py3/OpenCV/object_detection.py')
print(range(len(npd)))
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
import cv2
import numpy as np
import json
import os.path

with open("/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.json", "r") as f:
    json_data = json.load(f)

load_json_data = json_data
polygon_data = load_json_data["annotations"]

pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))



def xMax(xpoint):
    xmax = xpoint[0]
    for i in xpoint:
        if xmax < i:
            xmax = i
    return xmax


def yMax(ypoint):
    ymax = ypoint[0]
    for i in ypoint:
        if ymax < i:
            ymax = i
    return ymax

def xMin(xpoint):
    xmin = xpoint[0]
    for i in xpoint:
        if xmin > i:
            xmin = i
    return xmin

def yMin(ypoint):
    ymin = ypoint[0]
    for i in ypoint:
        if ymin > i:
            ymin = i
    return ymin


# cv2
img_path = "/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.jpg"
clicked_points = []
ird = cv2.imread(img_path)
reset = ird.copy();
points = []

xbox = []
ybox = []
xpoint = []
ypoint = []
for bucket in pd:
    xbox.append(bucket[::2])
    ybox.append(bucket[1::2])

findStr = "annotations" in load_json_data
classNum = 0;
if findStr:
    classNum = 0;


npd = np.array([pd[0], pd[1]], dtype=object)
print(range(len(npd)))
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
runcell(0, '/Users/daewon/.spyder-py3/OpenCV/object_detection.py')
import cv2
import numpy as np
import json
import os.path

filePath = "/Users/daewon/desktop/deep-i/python/PVC파이프/IMG_9046.json"
img_path = "/Users/daewon/desktop/deep-i/python/PVC파이프/IMG_9046.jpg"

with open(filePath, "r") as f:
    json_data = json.load(f)


load_json_data = json_data
polygon_data = load_json_data["annotations"]
findStr = "annotations" in load_json_data

classNum = 0;

if findStr:
    classNum = 0;

pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))

npd = np.array([pd[0], pd[1]], dtype=object)

def xMax(xpoint):
    xmax = xpoint[0]
    for i in xpoint:
        if xmax < i:
            xmax = i
    return xmax

def yMax(ypoint):
    ymax = ypoint[0]
    for i in ypoint:
        if ymax < i:
            ymax = i
    return ymax

def xMin(xpoint):
    xmin = xpoint[0]
    for i in xpoint:
        if xmin > i:
            xmin = i
    return xmin

def yMin(ypoint):
    ymin = ypoint[0]
    for i in ypoint:
        if ymin > i:
            ymin = i
    return ymin


# cv2
clicked_points = []
ird = cv2.imread(img_path)
reset = ird.copy();
points = []
print(ird.shape)
w, h = ird.shape
print(w)
print(w)
w, h = ird.shape
print(w)
w= ird.shape
print(w)
runcell(0, '/Users/daewon/.spyder-py3/OpenCV/object_detection.py')
blob = cv2.dnn.blobFromImage(ird, 0.00392, (416, 416), (0,0,0), True, crop=False)
print(blob)
import cv2
import numpy as np
import json
import os.path

filePath = "/Users/daewon/desktop/deep-i/python/PVC파이프/IMG_9046.json"
img_path = "/Users/daewon/desktop/deep-i/python/PVC파이프/IMG_9046.jpg"

with open(filePath, "r") as f:
    json_data = json.load(f)


load_json_data = json_data
polygon_data = load_json_data["annotations"]
findStr = "annotations" in load_json_data

classNum = 0;

if findStr:
    classNum = 0;

pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))

npd = np.array([pd[0], pd[1]], dtype=object)

def xMax(xpoint):
    xmax = xpoint[0]
    for i in xpoint:
        if xmax < i:
            xmax = i
    return xmax

def yMax(ypoint):
    ymax = ypoint[0]
    for i in ypoint:
        if ymax < i:
            ymax = i
    return ymax

def xMin(xpoint):
    xmin = xpoint[0]
    for i in xpoint:
        if xmin > i:
            xmin = i
    return xmin

def yMin(ypoint):
    ymin = ypoint[0]
    for i in ypoint:
        if ymin > i:
            ymin = i
    return ymin


# cv2
clicked_points = []
ird = cv2.imread(img_path)
reset = ird.copy();
points = []

# Detecting objects
blob = cv2.dnn.blobFromImage(ird, 0.00392, (416, 416), (0,0,0), True, crop=False)
print(blob)
print(blob)
print(ird.shape)
h, w, c = ird.shape
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
w = round(x1 - x2, 6)
rh = round(y1 - y2, 6)
xcenter = round(x2 + rw, 6)
ycenter = round(y2 + rh, 6)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(rw[0])
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
runcell(0, '/Users/daewon/.spyder-py3/OpenCV/object_detection.py')
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(lab[0])
lab = [(xcenter - rw) / 2, (ycenter - rh) / 2, xcenter + rw, ycenter + rh]
print(lab[0])
rw = round(x1 - x2, 6)
rh = round(y1 - y2, 6)
xcenter = round(rw/2 + x2, 6)
ycenter = round(rh/2 + y2, 6)
rows = ird.shape[0]
detection = [xcenter, ycenter, rw, rh]
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
runcell(0, '/Users/daewon/.spyder-py3/OpenCV/object_detection.py')

## ---(Mon Aug 22 16:14:43 2022)---
runcell(0, '/Users/daewon/.spyder-py3/OpenCV/object_detection.py')
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
textPath = " "
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
textPath = "test.txt"
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
textPath = "./test.txt"
data = "class xCenter  yCenter  w       h"
if os.path.isfile(textPath):
    f = open(textPath, "a", encoding = "UTF8")
else:
    f = open(textPath, "w", encoding = "UTF8")
    f.write(data)
textPath = "test.txt"
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(filePath1)
iter = glob.glob("./PVC파이프/*.json" , recursive=True)

for i in iter:
    print(i)
print(i)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
for i in iter:
    print(i)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(filePath1)
path_iter = []
for path in iter:
    path_iter.append(path)

print(path_iter)

file_iter = []
img_iter = []
for path in fiter:
    file_iter.append(path)

for path in iiter:
    img_iter.append(path)

print(img_iter)
fiter = glob.glob("./desktop/PVC파이프/*.json" , recursive=True)
iiter = glob.glob("./desktop/PVC파이프/*.jpg" , recursive=True)

file_iter = []
img_iter = []
for path in fiter:
    file_iter.append(path)

for path in iiter:
    img_iter.append(path)

print(img_iter)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(len(file_iter))
print(file_iter[0])
print(file_iter)
print(file_iter[len(file_iter)])
print(file_iter[len(file_iter) - 1])
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(file_iter[0])
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
src = "outoflange"
print(src)
print(str(file_iter[0]))
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print("%s" % file_iter[0])
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(npd)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(npd)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(npd)
print(npd[0])
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
for i in range(len(pd)):
    # npd = np.array([pd[0], pd[1]], dtype=object)
    npd = np.array(pd[i], dtype=object)

print(npd)
for i in range(len(npd)):
    xpoint = npd[i][::2]
    print(xpoint)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
for i in range(len(npd)):
    print(npd[i])
print(npd[i][::2])
for i in range(len(pd)):
    # npd = np.array([pd[0], pd[1]], dtype=object)
    npd = np.array(pd[i], dtype=object)

print(npd)
print(npd)
for i in range(len(pd)):
    # npd = np.array([pd[0], pd[1]], dtype=object)
    npd = np.array(pd[i-1], dtype=object)

print(npd)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(npd[i])
for i in range(len(pd)):
    # npd = np.array([pd[0], pd[1]], dtype=object)
    npd = np.array(pd[i-1], dtype=object)

print(npd[0])
print(npd)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
pd = []
npd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))


for i in range(len(pd)):
    # npd = np.array([pd[0], pd[1]], dtype=object)
    npd = np.append(npd, np.array(pd[i-1], dtype=object))

print(npd)
import cv2
import numpy as np
import json
import os.path
import glob

fiter = glob.glob("./desktop/PVC파이프/*.json" , recursive=True)
iiter = glob.glob("./desktop/PVC파이프/*.jpg" , recursive=True)

file_iter = []
img_iter = []
for path in fiter:
    file_iter.append(path)

for path in iiter:
    img_iter.append(path)


def Last2Yolo(filePath, imgPath):
    filePaths = filePath
    img_Paths = imgPath
    
    with open(filePaths, "r") as f:
        json_data = json.load(f)
    
    load_json_data = json_data
    polygon_data = load_json_data["annotations"]
    findStr = "annotations" in load_json_data
    
    classNum = 0;
    
    if findStr:
        classNum = 0;
    
    pd = []
    npd = []
    for bu1 in polygon_data:
        pd.append((bu1["polygon"]))
    
    
    for i in range(len(pd)):
        # npd = np.array([pd[0], pd[1]], dtype=object)
        npd = np.append(npd, np.array(pd[i-1], dtype=object))
    
    print(npd)
print(npd)
for i in range(len(pd)):
    # npd = np.array([pd[0], pd[1]], dtype=object)
    npd = np.append(npd, np.array(pd[i-1], dtype=object))
print(npd)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(npd)
for i in range(len(pd)):
    # npd = np.array([pd[0], pd[1]], dtype=object)
    npd = np.append(npd, np.array([pd[i-1]], dtype=object))
    print(npd)
print(npd)
for i in range(len(pd)):
    # npd = np.array([pd[0], pd[1]], dtype=object)
    npd = np.array(pd[i-1], dtype=object)

print(npd)
import cv2
import numpy as np
import json
import os.path
import glob

fiter = glob.glob("./desktop/PVC파이프/*.json" , recursive=True)
iiter = glob.glob("./desktop/PVC파이프/*.jpg" , recursive=True)

file_iter = []
img_iter = []
for path in fiter:
    file_iter.append(path)

for path in iiter:
    img_iter.append(path)


def Last2Yolo(filePath, imgPath):
    filePaths = filePath
    img_Paths = imgPath
    
    with open(filePaths, "r") as f:
        json_data = json.load(f)
    
    load_json_data = json_data
    polygon_data = load_json_data["annotations"]
    findStr = "annotations" in load_json_data
    
    classNum = 0;
    
    if findStr:
        classNum = 0;
    
    pd = []
    for bu1 in polygon_data:
        pd.append((bu1["polygon"]))
    
    
    for i in range(len(pd)):
        # npd = np.array([pd[0], pd[1]], dtype=object)
        npd = np.array(pd[i-1], dtype=object)
    
    print(npd)
print(npd)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(pd[0])
print(pd[1])
print(pd[0])
print(pd)
print(filePaths)
print(img_Paths)
import cv2
import numpy as np
import json
import os.path
import glob

fiter = glob.glob("./desktop/PVC파이프/*.json" , recursive=True)
iiter = glob.glob("./desktop/PVC파이프/*.jpg" , recursive=True)

file_iter = []
img_iter = []
for path in fiter:
    file_iter.append(path)

for path in iiter:
    img_iter.append(path)


def Last2Yolo(filePath, imgPath):
    filePaths = filePath
    img_Paths = imgPath
    
    print(filePaths)
    print(img_Paths)
print(filePaths)
print(img_Paths)
def Last2Yolo(filePath, imgPath):
    filePaths = filePath
    img_Paths = imgPath
    
    print(filePaths)
    print(img_Paths)
print(file_iter)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')

fiter = glob.glob("./desktop/PVC파이프/*.json" , recursive=True)
iiter = glob.glob("./desktop/PVC파이프/*.jpg" , recursive=True)

file_iter = []
img_iter = []
for path in fiter:
    file_iter.append(path)

for path in iiter:
    img_iter.append(path)

def Last2Yolo(filePath, imgPath):
    filePaths = filePath
    img_Paths = imgPath
    
    print(filePaths)
    print(img_Paths)
file_iter = []
img_iter = []
for path in fiter:
    file_iter.append(path)

for path in iiter:
    img_iter.append(path)

print(file_iter)
print(img_iter)
import cv2
import numpy as np
import json
import os.path
import glob

fiter = glob.glob("./desktop/PVC파이프/*.json" , recursive=True)
iiter = glob.glob("./desktop/PVC파이프/*.jpg")

file_iter = []
img_iter = []
for path in fiter:
    file_iter.append(path)

for path in iiter:
    img_iter.append(path)

print(file_iter)
print(img_iter)
import cv2
import numpy as np
import json
import os.path
import glob

fiter = glob.glob("./desktop/PVC파이프/*.json")
iiter = glob.glob("./desktop/PVC파이프/*.jpg")

file_iter = []
img_iter = []
for path in fiter:
    file_iter.append(path)

for path in iiter:
    img_iter.append(path)

print(file_iter)
print(img_iter)
for i in range(len(file_iter)):
    # Last2Yolo("%s" % (file_iter[i - 1]), "%s" % (img_iter[i - 1]))
    for j in range(len(img_iter)):
        findfilestr = file_iter[j - 1].split("_")
        print(findfilestr)
for j in range(len(img_iter)):
    findfilestr = file_iter[j - 1].split("_")
    print(findfilestr)
for j in range(len(img_iter)):
    findfilestr = file_iter[j - 1].split("_")
    print(findfilestr)
    print(findfilestr[0])
print(findfilestr[0])
print(file_iter[0])
import cv2
import numpy as np
import json
import os.path
import glob

fiter = glob.glob("./desktop/PVC파이프/*.json" , recursive=True)
iiter = glob.glob("./desktop/PVC파이프/*.jpg" , recursive=True)

file_iter = []
img_iter = []
for path in fiter:
    file_iter.append(path)

for path in iiter:
    img_iter.append(path)

for j in range(len(img_iter)):
    findfilestr = file_iter[j - 1].split("_")
    print(findfilestr)
    print(findfilestr[0])
for j in range(len(img_iter)):
    Fsstr = file_iter[j - 1].split("_")
    for k in range(len(Fsstr)):
        Ssstr = Fsstr[::2]

print(Ssstr)
for j in range(len(img_iter)):
    Fsstr = file_iter[j - 1].split("_")
    for k in range(len(Fsstr)):
        Ssstr = Fsstr[::2]

print(Fsstr)
for j in range(len(img_iter)):
    Fsstr = file_iter[j - 1].split("_")
    print(Fsstr)
for j in range(len(img_iter)):
    Fsstr = file_iter[j - 1].split("_")
    print(Fsstr[j][1])
for j in range(len(img_iter)):
    Fsstr = file_iter[j - 1].split("_")
    print(Fsstr[j - 1][1])
for j in range(len(file_iter)):
    Fsstr = file_iter[j - 1].split("_")
    print(Fsstr[j - 1][1])
for j in range(len(file_iter)):
    Fsstr = file_iter[j - 1].split("_")

print(Fsstr[0[1]])
print(Fsstr[0][1])
for j in range(len(file_iter)):
    Fsstr = file_iter[j - 1].split("_")

print(Fsstr[0][1])
for j in range(len(file_iter)):
    Fsstr = file_iter[j - 1].split("_")

print(Fsstr[0][0])
for j in range(len(file_iter)):
    Fsstr = file_iter[j - 1].split("_")
    print(Fsstr[0][0])
print(Fsstr[0][0])
print(Fsstr[0])
for j in range(len(file_iter)):
    Fsstr = file_iter[j - 1].split("_")
    print(Fsstr[1])
print(Fsstr[1])
print(Fsstr)
print(Fsstr[3])
print(file_iter)
print(file_iter[0])

for j in range(len(file_iter)):
    Fsstr = file_iter[j].split("_")
    Ssstr = Fsstr[1::2]
print(Ssstr)
print(Ssstr[0])
print(Ssstr[1])
Fsstr =[]
Ssstr = []
for j in range(len(file_iter)):
    Fsstr = file_iter[j].split("_")
    Ssstr = Fsstr[1::2]

print(Ssstr)
print(Fsstr)
for j in range(len(file_iter)):
    Fsstr = file_iter[j].split("_")
    Ssstr = Fsstr[1::2]

print(Fsstr)
Fsstr =[]
Ssstr = []
for j in range(len(file_iter)):
    Fsstr = file_iter[j].split("_")
    # Ssstr = Fsstr[1::2]

print(Fsstr)
print(file_iter[0])
print(file_iter[2])
print(file_iter[3])
print(file_iter[4])
for j in range(len(file_iter)):
    Fsstr = file_iter[j].split("_")
    # Ssstr = Fsstr[1::2]

print(Fsstr)
Fsstr =[]
Ssstr = []
for j in range(len(file_iter)):
    Fsstr = file_iter[j].split("_")
    print(Fsstr[j])
Fsstr =[]
Ssstr = []
for j in range(len(file_iter)):
    Fsstr = file_iter[j].split("_")
    print(Fsstr)
Fsstr =[]
Ssstr = []
for j in range(len(file_iter)):
    Fsstr = file_iter[j].split("_")
    Ssstr = Fsstr[1::2].split(".")
    print(Ssstr)
Fsstr =[]
Ssstr = []
for j in range(len(file_iter)):
    Fsstr = file_iter[j].split("_")
    Jsstr = Fsstr[1::2]
    print(Jsstr)
Fsstr =[]
Ssstr = []
for j in range(len(file_iter)):
    Fsstr = file_iter[j].split("_")
    Jsstr = Fsstr[1::2]
    
    print(Jsstr[j])
for i in range(len(Jsstr)):
    print(Jsstr[i])
Fsstr =[]
Ssstr = []
for i in range(len(file_iter)):
    Fsstr = file_iter[i].split("_")
    Jsstr = Fsstr[1::2]

for i in range(len(Jsstr)):
    print(Jsstr[i])
print(len(Jsstr))
import cv2
import numpy as np
import json
import os.path
import glob

fiter = glob.glob("./desktop/PVC파이프/*.json" , recursive=True)
iiter = glob.glob("./desktop/PVC파이프/*.jpg" , recursive=True)

file_iter = []
img_iter = []

for path in fiter:
    file_iter.append(path)

for path in iiter:
    img_iter.append(path)

Fsstr =[]
Ssstr = []
for i in range(len(file_iter)):
    Fsstr = file_iter[i].split("_")
    Jsstr = Fsstr[1::2]

print(len(Jsstr))
import cv2
import numpy as np
import json
import os.path
import glob

fiter = glob.glob("./desktop/PVC파이프/*.json" , recursive=True)
iiter = glob.glob("./desktop/PVC파이프/*.jpg" , recursive=True)

file_iter = []
img_iter = []

for path in fiter:
    file_iter.append(path)

for path in iiter:
    img_iter.append(path)

Fsstr =[]
Jsstr = []
for i in range(len(file_iter)):
    Fsstr = file_iter[i].split("_")
    Jsstr = Fsstr[1::2]

print(len(Jsstr))
Fsstr =[]
Jsstr = []
for i in range(len(file_iter)):
    Fsstr.append(file_iter[i].split("_"))
    Jsstr.append(Fsstr[1::2])

print(Fsstr)
print(len(Jsstr))
print(Jsstr)
Fsstr =[]
Jsstr = []
for i in range(len(file_iter)):
    Fsstr = file_iter[i].split("_")
    Jsstr.append(Fsstr[1::2])

print(Fsstr)
print(Jsstr)
print(Ssstr)
Fsstr =[]
Jsstr = []
Ssstr = []
for i in range(len(file_iter)):
    Fsstr = file_iter[i].split("_")
    Jsstr.append(Fsstr[1::2])
    Ssstr.append(Jsstr[i].split("."))

print(Ssstr)
Jsstr = []
for i in range(len(file_iter)):
    Fsstr = file_iter[i].split("_")
    Jsstr.append(Fsstr[1::2])
    Ssstr = Jsstr[i].split(".")
    print(Ssstr)
Jsstr = []
for i in range(len(file_iter)):
    Fsstr = file_iter[i].split("_")
    Jsstr.append(Fsstr[1::2])
    print(Jsstr[i])
print(type(file_iter))
print(Jsstr)
print(type(Jsstr))
JsonName = []
for i in range(len(Jsstr)) :
    Ssstr = Jsstr[i].split(".")
    JsonName.append(Ssstr)

print(JsonName)
print(type(Fsstr))
JsonName = []
for i in range(len(Jsstr)) :
    Ssstr = Jsstr[i].split(".")
    JsonName.append(Ssstr)

print(JsonName)
for i in range(len(Jsstr)) :
    print(type(Jsstr[0]))
for i in range(len(file_iter)):
    print(type(file_iter[0]))
JsonName = []
for i in range(len(Jsstr)) :
    Ssstr = str(Jsstr[i]).split(".")
    JsonName.append(Ssstr)
print(JsonName)
JsonName = []
for i in range(len(Jsstr)) :
    Ssstr = str(Jsstr[i]).split(".")
    JsonName.append(Ssstr)

print(JsonName[0])
Jsstr2 = []
JsonName = []
for i in range(len(Jsstr)) :
    Ssstr = str(Jsstr[i]).split(".")
    Jsstr2.append(Ssstr)
    JsonName.append(Jsstr2[i][0])

print(JsonName)
print(JsonName)
print(JsonName[0])
print(Jsstr2)
print(Ssstr)
print(Jsstr)
print(Jsstr[0])
Jsstr2 = []
JsonName = []
for i in range(len(Jsstr)) :
    Ssstr = str(Jsstr[i].split("."))
    print(Ssstr)
print(str(Jsstr[0]))
print(str(Jsstr[0]).split("."))
print(Jsstr[0])
print(str(Jsstr[0]))
print(Ssstr)
print(Trans)
Trans1 = str(Ssstr)
print(Trans1)
print(Jsstr)
for i in range(len(file_iter)):
    name, ext = os.path.splitext(file_iter)
    print("name:", name)
    print("ext:", ext)
for i in range(len(file_iter)):
    file_iter[i] = os.getcwd()
    name, ext = os.path.splitext(file_iter[i])
    print("name:", name)
    print("ext:", ext)
for i in range(len(file_iter)):
    
    
    basename = os.path.basename(file_iter[i])
    name, ext = os.path.splitext(basename)
    print("name:", name)
    print("ext:", ext)
basename = os.path.basename(file_iter[i])
print(basename)
print(fiter)
basename = os.path.basename(fiter[i])
print(basename)
basename = os.path.basename(fiter[i])
print(basename)
name, ext = os.path.splitext(basename)
print("name:", name)
print("ext:", ext)
for i in range(len(file_iter)):
    
    
    basename = os.path.basename(fiter[i])
    name, ext = os.path.splitext(basename)
    print("name:", name)
    print("ext:", ext)
Fname = []
for i in range(len(file_iter)):
    basename = os.path.basename(fiter[i])
    name, ext = os.path.splitext(basename)
    print("name:", name)
    print("ext:", ext)
    
    Fname.append(name)

print(Fname)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(img_iter)
print(Fname)
print(fiter)
for i in range(len(file_iter)):
    fPath = file_iter[i]
    for j in range(len(img_iter)):
        iPath = img_iter[j]
        for k in range(len(Fname)):
            if(Fname[k] in iPath):
                checkPath = iPath
    
    print(checkPath)
for i in range(len(file_iter)):
    fPath = file_iter[i]
    print(fPath)
    for j in range(len(img_iter)):
        iPath = img_iter[j]
        for k in range(len(Fname)):
            if(Fname[k] in iPath):
                checkPath = iPath
    
    print(checkPath)
fPath = file_iter[i]
print(fPath)
for j in range(len(img_iter)):
    iPath = img_iter[j]
    for k in range(len(Fname)):
        if(Fname[k] in iPath):
            checkPath = iPath

print(checkPath)
fPath = file_iter[0]
print(fPath)
for j in range(len(img_iter)):
    iPath = img_iter[j]
    for k in range(len(Fname)):
        if(Fname[k] in iPath):
            checkPath = iPath

print(checkPath)
fPath = file_iter[0]
print(fPath)
for j in range(len(img_iter)):
    iPath = img_iter[j]
    print(iPath)
    for k in range(len(Fname)):
        if(Fname[k] in iPath):
            checkPath = iPath

print(checkPath)
fPath = file_iter[0]
print(fPath)
for j in range(len(img_iter)):
    iPath = img_iter[j]
    print(iPath)
    for k in range(len(Fname)):
        print(Fname[k])
        if(Fname[k] in iPath):
            checkPath = iPath
fPath = file_iter[0]
print(fPath)
for j in range(len(img_iter)):
    iPath = img_iter[j]
    print(iPath)
    for k in range(len(Fname)):
        print(Fname[k])
        if(Fname[k] in iPath):
            checkPath = iPath
            print(checkPath)
fPath = file_iter[0]
print("파일 : " + fPath)
fPath = file_iter[0]
print("파일 : " + fPath)
for j in range(len(img_iter)):
    iPath = img_iter[j]
    print("이미지 경로 : " + iPath)
    for k in range(len(Fname)):
        print("파일 이름 : " + Fname[k])
        if(Fname[k] in iPath):
            checkPath = iPath
            print("확인 경로 : " + checkPath)
fPath = file_iter[0]
print("파일 : " + fPath)
for j in range(len(img_iter)):
    iPath = img_iter[j]
    print("이미지 경로 : " + iPath)
    if(Fname[i] in iPath):
        checkPath = iPath
        print("확인 경로 : " + checkPath)
fPath = file_iter[0]
print("파일 : " + fPath)
for j in range(len(img_iter)):
    iPath = img_iter[j]
    print("이미지 경로 : " + iPath)
    if(Fname[j] in iPath):
        checkPath = iPath
        print("확인 경로 : " + checkPath)
fPath = file_iter[0]
print("파일 : " + fPath)
for j in range(len(img_iter)):
    iPath = img_iter[j]
    print("이미지 경로 : " + iPath)
    if(Fname[0] in iPath):
        checkPath = iPath
        print("확인 경로 : " + checkPath)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
npd = []
for i in range(len(pd)):
    npd = np.array([pd[0], pd[1]], dtype=object)

print(npd)
import cv2
import numpy as np
import json
import os.path
import glob

fiter = glob.glob("./desktop/PVC파이프/*.json" , recursive=True)
iiter = glob.glob("./desktop/PVC파이프/*.jpg" , recursive=True)


file_iter = []
img_iter = []
for path in fiter:
    file_iter.append(path)

for path in iiter:
    img_iter.append(path)


Fname = []
for i in range(len(file_iter)):
    basename = os.path.basename(fiter[i])
    name, ext = os.path.splitext(basename) 
    Fname.append(name)

print(Fname)
print(fiter)    

def Last2Yolo(filePath, imgPath):
    filePaths = filePath
    img_Paths = imgPath
    
    with open(filePaths, "r") as f:
        json_data = json.load(f)
    
    load_json_data = json_data
    polygon_data = load_json_data["annotations"]
    findStr = "annotations" in load_json_data
    
    classNum = 0;
    
    if findStr:
        classNum = 0;
    
    pd = []
    for bu1 in polygon_data:
        pd.append((bu1["polygon"]))
    
    npd = []
    for i in range(len(pd)):
        npd = np.array([pd[0], pd[1]], dtype=object)
    
    print(npd)
print(npd)
print(pd)
runcell(0, '/Users/daewon/.spyder-py3/OpenCV/object_detection.py')
runcell(0, '/Users/daewon/.spyder-py3/OpenCV/object_detection.py')
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
import cv2
import numpy as np
import json
import os.path
import glob

fiter = glob.glob("./desktop/PVC파이프/*.json" , recursive=True)
iiter = glob.glob("./desktop/PVC파이프/*.jpg" , recursive=True)


file_iter = []
img_iter = []
for path in fiter:
    file_iter.append(path)

for path in iiter:
    img_iter.append(path)


Fname = []
for i in range(len(file_iter)):
    basename = os.path.basename(fiter[i])
    name, ext = os.path.splitext(basename) 
    Fname.append(name)


def Last2Yolo(filePath, imgPath):
    filePaths = filePath
    img_Paths = imgPath
    
    with open(filePaths, "r") as f:
        json_data = json.load(f)
    
    load_json_data = json_data
    polygon_data = load_json_data["annotations"]
    findStr = "annotations" in load_json_data
    
    classNum = 0;
    
    if findStr:
        classNum = 0;
    
    pd = []
    for bu1 in polygon_data:
        pd.append((bu1["polygon"]))
print(pd)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
npd = []
for i in range(len(pd)):
    npd = np.array(pd[i], dtype=object)

print(npd)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(type(npd))
print(type(npd[0][::2]))
print(npd)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(type(npd))
print(type(npd[0]))
print(npd[0])
print(npd)
print(pd)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print("pd : " + pd)
print(pd)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
import cv2
import numpy as np
import json
import os.path
import glob

fiter = glob.glob("./desktop/PVC파이프/*.json" , recursive=True)
iiter = glob.glob("./desktop/PVC파이프/*.jpg" , recursive=True)


file_iter = []
img_iter = []
for path in fiter:
    file_iter.append(path)

for path in iiter:
    img_iter.append(path)


Fname = []
for i in range(len(file_iter)):
    basename = os.path.basename(fiter[i])
    name, ext = os.path.splitext(basename) 
    Fname.append(name)


def Last2Yolo(filePath, imgPath):
    filePaths = filePath
    img_Paths = imgPath
    
    with open(filePaths, "r") as f:
        json_data = json.load(f)
    
    load_json_data = json_data
    polygon_data = load_json_data["annotations"]
    findStr = "annotations" in load_json_data
    
    classNum = 0;
    
    if findStr:
        classNum = 0;
    
    pd = []
    for bu1 in polygon_data:
        pd.append((bu1["polygon"]))
    
    print(pd)
    
    npd = []
    for i in range(len(pd)):
        npd = np.array(pd[i], dtype=object)
    
    print(npd)
    print(type(npd[0]))
print(pd[1])
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print("npd : " + npd)
print(type(npd[0]))
print(npd)
print(type(npd[0]))
npd = []
for i in range(len(pd)):
    npd = np.array(pd[i-1], dtype=object)

print(npd)
print(type(npd[0]))
npd = []
for i in range(len(pd)):
    #npd = np.array(pd[i-1], dtype=object)
    npd.append(pd[i])

print(npd)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
import cv2
import numpy as np
import json
import os.path
import glob

fiter = glob.glob("./desktop/consquare/**/*.json" , recursive=True)
iiter = glob.glob("./desktop/consquare/**/*.jpg" , recursive=True)

# 강관파이프 = 0, 거푸집 = 1, 단열재 = 2, 동바리 = 3, 라바콘 = 4, 벽돌 = 5, 부직포 = 6, 스터드 = 7, 스틸그레이팅 = 8
# 외벽석자재 = 9, 외장석자재 = 10, 철근 = 11, 토류판 = 12, 파이프보온재 = 13, 풀륨관 = 14, 흄관 = 15, 창호 = 16
# PCV파이프 = 17

material = ["강관파이프", "거푸집", "단열재", "동바로", "라바콘", "벽돌", "부직포", "스터드",
            "스틸그레이팅", "외벽석자재", "출근", "토류판", "파이프보온재", "풀륨관", "흄관", "창호", "PCV파이프"]

file_iter = []
img_iter = []
for path in fiter:
    file_iter.append(path)

for path in iiter:
    img_iter.append(path)

Fname = []
for i in range(len(file_iter)):
    basename = os.path.basename(fiter[i])
    name, ext = os.path.splitext(basename) 
    Fname.append(name)

def Last2Yolo(filePath, imgPath):
    filePaths = filePath
    img_Paths = imgPath
    
    with open(filePaths, "r") as f:
        json_data = json.load(f)
    
    load_json_data = json_data
    polygon_data = load_json_data["annotations"]
#    findStr = "annotations" in load_json_data
    
    className = []
    for cn in polygon_data:
        className.append(cn["class"])
    
    for i in range(len(material)):
        if (className == material[i]):
            classNum = i
    print(classNum)
print(classNum)
className = []
for cn in polygon_data:
    className.append(cn["class"])
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(len(material))
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(len(material))
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
for i in range(len(material)):
    print(material[i])
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
className = []
for cn in polygon_data:
    className.append(cn["class"])

for i in range(len(material)):
    if (className[0] == material[i]):
        classNum = i
className = []
for cn in polygon_data:
    className.append(cn["class"])
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(className)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
import cv2
import numpy as np
import json
import os.path
import glob
import os

material = ["강관파이프", "거푸집", "단열재", "동바리", "라바콘", "벽돌", "부직포", "스터드",
            "스틸그레이팅", "외벽석자재", "외장석자재", "철근", "토류판", "파이프보온재", "풀륨관", "흉관", "창호", "PVC파이프"]

fiter = glob.glob("./desktop/consquare/**/*.json" , recursive=True)
iiter = glob.glob("./desktop/consquare/**/*.jpg" , recursive=True)

mkdir_path = "./desktop/cpdir/"
os.mkdir(mkdir_path)

in_dir = []
for i in range(len(material)):
    in_dir.append(mkdir_path + material[i])
    os.mkdir(in_dir)
runcell(0, '/Users/daewon/Desktop/deep-i/python/sub2.py')
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(Fname)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(len(file_iter))
print(i)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
for i in  range(len(material)):
    select_path = "./desktop/consquare/" + material[i] + "/*.jpg"
    file_list = os.listdir(select_path)
    file_list_jpg = [file for file in file_list if file.endswith(".jpg")]
    
    print(file_list_jpg)
for i in  range(len(material)):
    select_path = "./desktop/consquare/" + material[i] + "/*.jpg"
    print(select_path)
    file_list = os.listdir(select_path)
    file_list_jpg = [file for file in file_list if file.endswith(".jpg")]
    
    print(file_list_jpg)
for i in  range(len(material)):
    select_path = "./desktop/consquare/" + material[i]
    print(select_path)
    file_list = os.listdir(select_path)
    file_list_jpg = [file for file in file_list if file.endswith(".jpg")]
    
    print(file_list_jpg)
for i in  range(len(material)):
    select_path = "./desktop/consquare/" + material[i] + "/"
    print(select_path)
    file_list = os.listdir(select_path)
    file_list_jpg = [file for file in file_list if file.endswith(".jpg")]
    
    print(file_list_jpg)
import cv2
import numpy as np
import json
import os.path
import glob
import os
import shutil

# 강관파이프 = 0, 거푸집 = 1, 단열재 = 2, 동바리 = 3, 라바콘 = 4, 벽돌 = 5, 부직포 = 6, 스터드 = 7, 스틸그레이팅 = 8
# 외벽석자재 = 9, 외장석자재 = 10, 철근 = 11, 토류판 = 12, 파이프보온재 = 13, 풀륨관 = 14, 흉관 = 15, 창호 = 16
# PVC파이프 = 17
material = ["강관파이프", "거푸집", "단열재", "동바리", "라바콘", "벽돌", "부직포", "스터드",
            "스틸그레이팅", "외벽석자재", "외장석자재", "철근", "토류판", "파이프보온재", "풀륨관", "흉관", "창호", "PVC파이프"]

fiter = glob.glob("./desktop/consquare/**/*.json" , recursive=True)
iiter = glob.glob("./desktop/consquare/**/*.jpg" , recursive=True)

mkdir_path = "./desktop/copycon/"
os.makedirs(mkdir_path, exist_ok=True)

#in_dir = ""
#for i in range(len(material)):
#    in_dir = mkdir_path + material[i]
#    os.makedirs(in_dir, exist_ok=True)


file_iter = []
img_iter = []
for path in fiter:
    file_iter.append(path)

for path in iiter:
    img_iter.append(path)

Fname = []
for i in range(len(file_iter)):
    basename = os.path.basename(fiter[i])
    name, ext = os.path.splitext(basename) 
    Fname.append(name)


def Last2Yolo(filePath, imgPath, fio):
    filePaths = filePath
    img_Paths = imgPath
    
    with open(filePaths, "r") as f:
        json_data = json.load(f)
    
    load_json_data = json_data
    polygon_data = load_json_data["annotations"]
    
    className = ""
    for cn in polygon_data:
        className = cn["class"]
    
    classNum = 0
    for i in range(len(material)):
        if (className == material[i]):
            classNum = i
    
    pd = []
    for bu1 in polygon_data:
        pd.append((bu1["polygon"]))
    
    npd = []
    for i in range(len(pd)):
        npd.append(pd[i])    
    
    def xMax(xpoint):
        xmax = xpoint[0]
        for i in xpoint:
            if xmax < i:
                xmax = i
        return xmax
    
    def yMax(ypoint):
        ymax = ypoint[0]
        for i in ypoint:
            if ymax < i:
                ymax = i
        return ymax
    
    def xMin(xpoint):
        xmin = xpoint[0]
        for i in xpoint:
            if xmin > i:
                xmin = i
        return xmin
    
    def yMin(ypoint):
        ymin = ypoint[0]
        for i in ypoint:
            if ymin > i:
                ymin = i
        return ymin
    
    # cv2
    ird = cv2.imread(img_Paths)
    h, w, c = ird.shape
    
    xpoint = []
    ypoint = []
    
    # 한 txt에 저장
    # textPath = "./dir/.txt"
    for i in range(len(npd)):    
        xpoint = npd[i][::2]
        ypoint = npd[i][1::2]
        
        x1 = xMax(xpoint) / w
        y1 = yMax(ypoint) / h
        x2 = xMin(xpoint) / w
        y2 = yMin(ypoint) / h
        
        rw = format(x1 - x2, ".6f")#round(x1 - x2, 6)
        rh = format(y1 - y2, ".6f")#round(y1 - y2, 6)
        xcenter = format(float(rw) / 2 + x2, ".6f")#round(rw/2 + x2, 6)
        ycenter = format(float(rh) / 2 + y2, ".6f")#round(rh/2 + y2, 6)
        
        fio.write(str(classNum) + " " + str(xcenter) + " " + str(ycenter) + " " + str(rw) + " " + str(rh))
    
    f.close()

for i in range(len(file_iter)):
    fPath = file_iter[i]
    textPath = "./desktop/stext/" + Fname[i]+".txt"
    if os.path.isfile(textPath):
        fio = open(textPath, "a", encoding = "UTF8")
    else:
        fio = open(textPath, "w", encoding = "UTF8")
    
    for j in range(len(img_iter)):
        iPath = img_iter[j]
        if(Fname[i] in iPath):
            checkPath = iPath   
    Last2Yolo(fPath, checkPath, fio)

shutil.copytree("./desktop/consquare", "./desktop/copycon")
for i in  range(len(material)):
    select_path = "./desktop/consquare/" + material[i] + "/"
    file_list = os.listdir(select_path)
    file_list_jpg = [file for file in file_list if file.endswith(".jpg")]
shutil.copytree("./desktop/consquare", "./desktop/copycon")
for i in  range(len(material)):
    select_path = "./desktop/consquare/" + material[i] + "/"
    file_list = os.listdir(select_path)
    file_list_jpg = [file for file in file_list if file.endswith(".jpg")]
mkdir_path = "./desktop/copycon/"
os.makedirs(mkdir_path, exist_ok=True)
shutil.copytree("./desktop/consquare", "./desktop/copycon")
rmtxt = glob.glob("./desktop/copycon/*.json")

print(len(rmtxt))
rmtxt = glob.glob("./desktop/copycon/**/*.json")

print(len(rmtxt))
for i in range(len(rmjson)):
    if os.path.exists("./desktop/copycon/**/*.json") :
        os.remove(rmjson[i])
rmjson = glob.glob("./desktop/copycon/**/*.json")


for i in range(len(rmjson)):
    if os.path.exists("./desktop/copycon/**/*.json") :
        os.remove(rmjson[i])
for i in range(len(rmjson)):
    if os.path.exists("./desktop/copycon/**/*.json") :
        os.remove(rmjson[i])
runcell(0, '/Users/daewon/Desktop/deep-i/python/sub2.py')
rmjson = glob.glob("./desktop/copycon/**/*.json")
for i in range(len(rmjson)):
    if os.path.exists("./desktop/copycon/**/*.json") :
        os.remove(rmjson[i])
runcell(0, '/Users/daewon/Desktop/deep-i/python/sub2.py')
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
rmjson = glob.glob("./desktop/copycon/**/*.json")

#for i in  range(len(material)):
#    select_path = "./desktop/consquare/" + material[i] + "/"
#    file_list = os.listdir(select_path)
#    file_list_jpg = [file for file in file_list if file.endswith(".jpg")]

for i in range(len(rmjson)):
    os.remove(rmjson[i])
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
findfile = glob.glob("./desktop/copycon/**/*.txt")
for i in range(len(Fname)):
    movefileDir = "./desktop/copycon/"
    throwfile = Fname[i] + ".txt"
    for j in range(len(material)):
        movedfileDir = movefileDir + material[i]
        shutil.move(movefileDir + throwfile, movedfileDir + throwfile)
findfile = glob.glob("./desktop/copycon/**/*.txt")
for i in range(len(Fname)):
    movefileDir = "./desktop/copycon/"
    throwfile = Fname[i] + ".txt"
    for j in range(len(material)):
        movedfileDir = movefileDir + material[j]
        shutil.move(movefileDir + throwfile, movedfileDir + throwfile)
for i in range(len(Fname)):
    movefileDir = "./desktop/copycon/"
    throwfile = Fname[i] + ".txt"
    for j in range(len(material)):
        movedfileDir = movefileDir + material[j]
        shutil.move(movefileDir + throwfile, movedfileDir + throwfile)
findfile = glob.glob("./desktop/copycon/**/*.txt")
for i in range(len(findfile)):
    movefileDir = "./desktop/copycon/"
    throwfile = Fname[i] + ".txt"
    for j in range(len(material)):
        movedfileDir = movefileDir + material[j]
        shutil.move(movefileDir + throwfile, movedfileDir + throwfile)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
runcell(0, '/Users/daewon/Desktop/deep-i/python/sub2.py')
movedfileDir = movefileDir + material[j] + "/"
print(movedfileDir)
runcell(0, '/Users/daewon/Desktop/deep-i/python/sub2.py')
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(Fname)
runcell(0, '/Users/daewon/Desktop/deep-i/python/sub2.py')
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
print(outCn)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
shutil.copy("./desktop/copycon/20220630_163158.txt", "./desktop/copycon/스틸그레이팅")
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
shutil.copy("./desktop/copycon/20220630_163158.txt", "./desktop/copycon/스틸그레이팅")
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')
shutil.copy("./desktop/copycon/20220630_163158.txt", "./desktop/copycon/스틸그레이팅")
shutil.copy(textPath, textPath2)
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')