import cv2
import numpy as np
import json
import os.path


filePath = "/Users/daewon/desktop/PVC파이프/IMG_9046.json"
imgPath = "/Users/daewon/desktop/PVC파이프/IMG_9046.json"

filePath = "/Users/daewon/desktop/PVC파이프/IMG_9046.json"
img_path = "/Users/daewon/desktop/PVC파이프/IMG_9046.json"

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
ird = cv2.imread(img_path)

xbox = []
ybox = []
xpoint = []
ypoint = []

for bucket in pd:
    xbox.append(bucket[::2])
    ybox.append(bucket[1::2])
    
textPath = "/Users/daewon/desktop/test.txt"
data = "class xMin yMin xMax yMax"
if os.path.isfile(textPath):
    f = open(textPath, "a", encoding = "UTF8")
else:
    f = open(textPath, "w", encoding = "UTF8")
    f.write(data)

for i in range(len(npd)):
    #xpoint = xbox[i]
    #ypoint = ybox[i]
    xpoint = npd[i][::2]#xbox[i]
    ypoint = npd[i][1::2]#ybox[i]

    x1 = xMax(xpoint)
    y1 = yMax(ypoint)
    x2 = xMin(xpoint)
    y2 = yMin(ypoint)
      
    img = cv2.rectangle(ird, (x2, y2), (x1, y1), (255, 0, 0), 7)
    f.write("\n" + str(classNum) + "     " + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2))

f.close()






