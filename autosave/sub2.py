#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:57:48 2022

@author: daewon
"""
print("pd"  1)

# class x y w h

0     21 23 45 61
1     21 23 45 61
2     21 23 45 61


poly = []
for data in polygon_data:
    poly.append(data["polygon"])
          
# 원본 파일을 가져 와서 clicked_points에 있는 점들을 그린다. , point[0] = x, point[1] = y
for point in poly:
    cv2.circle(ird, (point[0], point[1]), 2, (0, 255, 0), thickness=20)


poly_np = np.array(poly, dtype = np.int32)


cv2.imshow("image", ird)

key = cv2.waitKey()
if key == 13:
    cv2.polylines(ird, [poly_np], True, (0,255,0), 4)
    #cv2.fillPoly(ird, [fpoints], (0,255,0))
    cv2.imshow("image", ird)


