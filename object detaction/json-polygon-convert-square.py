import cv2
import numpy as np
import json
import os.path
import glob
import os
import shutil
import time     

# 강관파이프 = 0, 거푸집 = 1, 단열재 = 2, 동바리 = 3, 라바콘 = 4, 벽돌 = 5, 부직포 = 6, 스터드 = 7, 스틸그레이팅 = 8
# 외벽석자재 = 9, 외장석자재 = 10, 철근 = 11, 토류판 = 12, 파이프보온재 = 13, 풀륨관 = 14, 흉관 = 15, 창호 = 16
# PVC파이프 = 17
material = ["강관파이프", "거푸집", "단열재", "동바리", "라바콘", "벽돌", "부직포", "스터드",
            "스틸그레이팅", "외벽석자재", "외장석자재", "철근", "토류판", "파이프보온재", "풀륨관", "흉관", "창호", "PVC파이프"]

fiter = glob.glob("./desktop/consquare/**/*.json" , recursive=True)
iiter = glob.glob("./desktop/consquare/**/*.jpg" , recursive=True)

shutil.copytree("./desktop/consquare", "./desktop/copycon")
rmjson = glob.glob("./desktop/copycon/**/*.json")

outCn = 0
for i in range(len(rmjson)):
    os.remove(rmjson[i])
        
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
    global outCn

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
            outCn = i
            print(outCn) 
    
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
        
        fio.write(str(classNum) + " " + str(xcenter) + " " + str(ycenter) + " " + str(rw) + " " + str(rh) + "\n")

    f.close()
    
    textPath2 = "./desktop/copycon/" + material[outCn] + "/"
    # shutil.copy2(textPath, textPath2)  
    shutil.move(textPath, textPath2)

for i in range(len(file_iter)):
    fPath = file_iter[i]
    textPath = "./desktop/copycon/" + Fname[i] + ".txt"
    
    if os.path.isfile(textPath):
        fio = open(textPath, "a", encoding = "UTF8")
    else:
        fio = open(textPath, "w", encoding = "UTF8")
                
    for j in range(len(img_iter)):
        iPath = img_iter[j]
        if(Fname[i] in iPath):
            checkPath = iPath   
    Last2Yolo(fPath, checkPath, fio)
    
    # findfile = glob.glob("./desktop/copycon/*.txt")
    #for i in range(len(findfile)):
    
     #shutil.copy("./desktop/copycon/20220630_163158.txt", "./desktop/copycon/스틸그레이팅")   

                # shutil.move(os.path.join(movefileDir, os.path.basename(throwfile)), os.path.join(movedfileDir, throwfile))
                
            

# findfile = glob.glob("./desktop/copycon/*.txt")
# for i in range(len(findfile)):
#     movefileDir = "./desktop/copycon/"
#     throwfile = Fname[i] + ".txt"
#     for j in range(len(material)):
#         movedfileDir = movefileDir + material[j]
#         shutil.move(movefileDir + throwfile, movedfileDir + throwfile)