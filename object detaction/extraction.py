# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 20:05:49 2022

@author: khb05
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np, json, os.path, glob, os, cv2
import pandas as pd


class_list = ["A1_구진_플라크", "A2_비듬_각질_상피성잔고리", "A3_태선화_과다색소침착", "A4_농포_여드림", "A5_미란_궤양", "A6_결절_종괴"]

# current_path = os.getcwd();
# jsnpaths = glob.glob("C:\\Users\\khb05\\바탕 화면\\data_sample\\**\\*.json")
# imgpaths = glob.glob("C:\\Users\\khb05\\바탕 화면\\data_sample\\**\\*.jpg")
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
                
        txtname = jsnpaths[i].split("\\")[6].replace(".json", ".txt") 
        bbox.append([class_list.index(class_name), format(x, ".6f"), format(y, ".6f"), format(w, ".6f"), format(h, ".6f")])
                
        df = pd.DataFrame(bbox)
        df.to_csv("./" + class_name + "/" + txtname, header = None, index = False, sep=' ')
        

       
       
       
    
    
    
    



