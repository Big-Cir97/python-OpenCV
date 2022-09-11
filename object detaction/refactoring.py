import numpy as np
import json
import os.path
import glob
import os
import shutil
import cv2
import pandas as pd

# 강관파이프 = 0, 거푸집 = 1, 단열재 = 2, 동바리 = 3, 라바콘 = 4, 벽돌 = 5, 부직포 = 6, 스터드 = 7, 스틸그레이팅 = 8
# 외벽석자재 = 9, 외장석자재 = 10, 철근 = 11, 토류판 = 12, 파이프보온재 = 13, 풀륨관 = 14, 흉관 = 15, 창호 = 16
# PVC파이프 = 17
material = ["강관파이프", "거푸집", "단열재", "동바리", "라바콘", "벽돌", "부직포", "스터드",
            "스틸그레이팅", "외벽석자재", "외장석자재", "철근", "토류판", "파이프보온재", "풀륨관", "흉관", "창호", "PVC파이프"]

current_path = os.getcwd()
copy_path = os.mkdir(current_path + "/copycon/")

for i in range(len(material)):
    os.mkdir(current_path + "/copycon/" + material[i])
    
for i in range(len(material)) :
    json_files = glob.glob("./consquare/" + material[i] + "/*.json")
    new_path = "./copycon/" + material[i]

    for j in json_files:
        filename = j.split('/')[3]
        shutil.copy2(j, new_path)    
        filename1 = filename.replace('.json','.jpg')
        shutil.copy2("./consquare/" + material[i] + "/" + filename1, new_path)
        
    
json_read = glob.glob("./copycon/*/*.json")
img_path = glob.glob("./copycon/*/*.jpg")

  
def TransYoloForm(img_path, json_read, i):
    ird = cv2.imread(img_path[i])
    h, w, c = ird.shape
    
    with open(json_read[i], "r") as f:
        json_data = json.load(f)
    
    load_json_data = json_data
    polygon_data = load_json_data["annotations"]
    
    bbox = []
    for k in polygon_data:

        xpoint = k["polygon"][::2]
        ypoint = k["polygon"][1::2]
        
        xMax = max(xpoint)
        xMin = min(xpoint)
        yMax = max(ypoint)
        yMin = min(ypoint)
        
        rw = format((xMax / w) - (xMin / w), ".6f")
        rh = format((yMax / h) - (yMin / h), ".6f")
        
        xcenter = format(float(rw) / 2 + (xMin / w), ".6f")
        ycenter = format(float(rh) / 2 + (yMin / h), ".6f")
        
        if k["class"] == "비계" :
            continue
        bbox.append([material.index(k["class"]),xcenter,ycenter,rw,rh])
        
        
    
    copy_json_name = json_read[i].split("/")[3]
    direct_name = json_read[i].split("/")[2]
    txtname = copy_json_name.replace(".json", ".txt") 
    
    print("text write")
    
    df = pd.DataFrame(bbox)
    df.to_csv("./copycon/" + direct_name + "/" + txtname, header=None, index=False, sep='\t')
    

for i in range(len(json_read)):
    TransYoloForm(img_path, json_read, i)
    
    
# remove json
# rmjson = glob.glob("./desktop/copycon/**/*.json")
# for i in range(len(rmjson)):
#     os.remove(rmjson[i])   
    
    

    