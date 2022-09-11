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

## ---(Thu Aug 25 14:37:37 2022)---
runcell(0, '/Users/daewon/Desktop/deep-i/python/main.py')

## ---(Mon Aug 29 13:29:29 2022)---
runcell(0, '/Users/daewon/Desktop/deep-i/python/refactoring.py')
import numpy as np
import json
import os.path
import glob
import os
import shutil
import cv2
import pandas as pd
import cv2
runcell(0, '/Users/daewon/Desktop/deep-i/python/refactoring.py')

## ---(Tue Aug 30 14:56:43 2022)---
runcell(0, '/Users/daewon/Desktop/deep-i/python/refactoring.py')

## ---(Fri Sep  2 11:18:03 2022)---
material = ["강관파이프", "거푸집", "단열재", "동바리", "라바콘", "벽돌", "부직포", "스터드",
            "스틸그레이팅", "외벽석자재", "외장석자재", "철근", "토류판", "파이프보온재", "풀륨관", "흉관", "창호", "PVC파이프"]