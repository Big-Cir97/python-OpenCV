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
    
findfile = glob.glob("./desktop/copycon/*.txt")
for i in range(len(findfile)):
    movefileDir = "./desktop/copycon/"
    throwfile = Fname[i] + ".txt"
    for j in range(len(material)):
        movedfileDir = movefileDir + material[j] + "/"
        shutil.move(movefileDir + throwfile, movedfileDir)
        
        
shutil.copy("./desktop/copycon/20220630_163158.txt", "./desktop/copycon/스틸그레이팅")        
# print("pd"  1)

# class x y w h

# 0     21 23 45 61
# 1     21 23 45 61
# 2     21 23 45 61


# poly = []
# for data in polygon_data:
#     poly.append(data["polygon"])
          
# # 원본 파일을 가져 와서 clicked_points에 있는 점들을 그린다. , point[0] = x, point[1] = y
# for point in poly:
#     cv2.circle(ird, (point[0], point[1]), 2, (0, 255, 0), thickness=20)


# poly_np = np.array(poly, dtype = np.int32)


# cv2.imshow("image", ird)

# key = cv2.waitKey()
# if key == 13:
#     cv2.polylines(ird, [poly_np], True, (0,255,0), 4)
#     #cv2.fillPoly(ird, [fpoints], (0,255,0))
#     cv2.imshow("image", ird)

# shutil.copytree("./desktop/consquare", "./desktop/copycon")
# rmjson = glob.glob("./desktop/copycon/**/*.json")
# for i in range(len(rmjson)):
#     os.remove(rmjson[i])
    #if os.path.exists("./desktop/copycon/**/*.json") :
        
        
