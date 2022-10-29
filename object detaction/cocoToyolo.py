#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np, json, os.path, glob, os, shutil, cv2
import pandas as pd


class_list = ["A1_구진_플라크", "A2_비듬_각질_상피성잔고리", "A3_태선화_과다색소침착", "A4_농포_여드림", "A5_미란_궤양", "A6_결절_종괴"]

jsnpaths = glob.glob("/Users/daewon/desktop/IMG_D_A1_000001.json")
imgpaths = glob.glob("/Users/daewon/desktop/123.jpeg")


for i in range(len(jsnpaths)) :
    bbox = []
    with open(jsnpaths[i], "r") as f:
        load_json_data = json.load(f)
        
    ird = cv2.imread(imgpaths[i])   
    img_height, img_width, _ = ird.shape
    
    
    for k in range(len(load_json_data["labelingInfo"])) :
        if (list(load_json_data["labelingInfo"][k].keys())[0] == "polygon") :
            continue
        
        box_data = load_json_data["labelingInfo"][k]["box"]["location"][0]
        x, y, w, h = box_data["x"], box_data["y"], box_data["width"], box_data["height"]
        class_name = load_json_data["labelingInfo"][0]["polygon"]["label"]
    
        dw = 1.0 / img_width
        dh = 1.0 / img_height
        x_center = x + w / 2.0
        y_center = y + h / 2.0
        
        x = x_center * dw
        y = y_center * dh
        w = w * dw
        h = h * dh
    
        txtname = jsnpaths[i].split("/")[4].replace(".json", ".txt") 
        bbox.append([class_list.index(class_name), x, y, w, h])
    
        df = pd.DataFrame(bbox)
        df.to_csv(txtname, header = None, index = False, sep=' ')
        
    # [1]["box"]["location"][0]["x"]
   
    # C:/Users/oks19/Desktop/hyebin/h_bin/152.반려동물 피부질환 데이터/01.데이터/1.Training/2_라벨링데이터/TL01/반려견/피부/일반카메라/유증상/" + class_name + "/" + txtname, header = None, index = False, sep=' ')
    
    



