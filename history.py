
runfile('/Users/daewon/.spyder-py3/OpenCV/timecap.py', wdir='/Users/daewon/.spyder-py3/OpenCV')

## ---(Tue Aug 16 14:31:33 2022)---
runfile('/Users/daewon/Desktop/d/temp.py', wdir='/Users/daewon/Desktop/d')
runcell('테스트 FLASK API (데이터 전송 체크)', '/Users/daewon/Desktop/d/temp.py')
import requests

# 저장 이미지
img = '/Users/daewon/desktop/d/file.png'
f = open(img, 'rb')
files = {"file": (img, f)}

res = requests.post('http://43.200.14.101:7000/file',
                    files = files,
                    data={"C_ID": "deepi.contact.us@gmail.com",
                    "DATE": 2021051806,   # 전송시점 시간 (MCU 기준 시간 사용)
                    "TEMP": 32.5,           # 온도 센서값
                    "PH"  : 125.25,         # PH 센서값 
                    "SALT": 21.2158,        # 염도 센서값
                    "TURB": 1.212644,       # 탁도 센서값 
                    "DO"  : 21.2121542,     # 산소 포화도 센서값  
                    "IMG" : "수집양식장코드_수집일자.jpg"})
                                            # 서버에 저장되는 이미지 경로
                                            # API 2단 구성     
                                            # 1. IMG 전송 -> 이미지 경로값 응답
                                            # 2. 경로값 + 센서값 전송
f = open(img, 'rb')
files = {"file": (img, f)}

res = requests.post('http://43.200.14.101:7000/file',
                    files = files,
                    data={"C_ID": "deepi.contact.us@gmail.com",
                    "DATE": 2021051806,   # 전송시점 시간 (MCU 기준 시간 사용)
                    "TEMP": 32.5,           # 온도 센서값
                    "PH"  : 125.25,         # PH 센서값 
                    "SALT": 21.2158,        # 염도 센서값
                    "TURB": 1.212644,       # 탁도 센서값 
                    "DO"  : 21.2121542,     # 산소 포화도 센서값  
                    "IMG" : "수집양식장코드_수집일자.jpg"})
                                            # 서버에 저장되는 이미지 경로
                                            # API 2단 구성     
                                            # 1. IMG 전송 -> 이미지 경로값 응답
                                            # 2. 경로값 + 센서값 전송


f.close()
runcell('테스트 FLASK API (데이터 전송 체크)', '/Users/daewon/Desktop/d/temp.py')

## ---(Tue Aug 16 16:50:54 2022)---
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
%clear
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
runfile('/Users/daewon/.spyder-py3/temp.py', wdir='/Users/daewon/.spyder-py3')
runcell(0, '/Users/daewon/.spyder-py3/temp.py')

## ---(Wed Aug 17 14:04:12 2022)---
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
%clear
runcell(0, '/Users/daewon/.spyder-py3/temp.py')

## ---(Wed Aug 17 15:06:10 2022)---
runcell(0, '/Users/daewon/.spyder-py3/temp.py')

## ---(Thu Aug 18 10:33:54 2022)---
runcell(0, '/Users/daewon/Desktop/jsontest.py')
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
runcell(0, '/Users/daewon/Desktop/jsontest.py')
runcell(0, '/Users/daewon/.spyder-py3/temp.py')

for data in json_data:
    print(data)
json_data['coordinate']

for data in json_data['coordinate']:
    print(data)
data['x']
int(data['x'])
poly = []
for data in json_data['coordinate']:
    poly.append( [int(data['x']),int(data['y']])

poly = []
for data in json_data['coordinate']:
    poly.append([int(data['x']),int(data['y']]))
poly = []
for data in json_data['coordinate']:
    poly.append([int(data['x']),int(data['y'])])
poly = []
for data in json_data['coordinate']:
    poly.append(int(data['x']),int(data['y']))
pdata = np.array[data]
print(data)
for data in json_data['coordinate']:
    poly.append([int(data['x']), int(data['y'])])
poly.append([int(data['x']), int(data['y'])])
for data in json_data['coordinate']:
    poly.append([int(data['x']), int(data['y'])])
print(json_data)
print(json_data["coordinate"])
for data in json_data['coordinate']:
    poly.append([int(data['x']), int(data['y'])])
for data in json_data['coordinate'].legnth():
    poly.append([int(data['x']), int(data['y'])])
for data in json_data['coordinate'].length():
    poly.append([int(data['x']), int(data['y'])])
poly = []
for data in json_data['coordinate'].len():
    poly.append([int(data['x']), int(data['y'])])
print(json_data["coordinate"])
for data in json_data['coordinate']:
    poly.append([int(data['x']), int(data['y'])])
print(poly)
poly_np = np.array(poly, dtype = np.int32)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
print(point)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
print(point)
print(data)
for data in json_data['coordinate']:
    poly.append([int(data['x']), int(data['y'])])
    print(data)
print(data)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
print(data)
for data in json_data['coordinate']:
    poly.append([int(data['x']), int(data['y'])])
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
for data in json_data['coordinate']:
    poly.append([int(data['x']), int(data['y'])])
with open("/Users/daewon/desktop/deep-i/python/PVC파이프/poly.json", "r") as f:
    json_data = json.load(f)

# print(json_data)
# print(json_data["coordinate"][0]["x"])

# cv2
img_path = "/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_152428.jpg"
clicked_points = []
ird = cv2.imread(img_path)
reset = ird.copy();
points = []
count = 0;

poly = []
for data in json_data['coordinate']:
    poly.append([int(data['x']), int(data['y'])])
import cv2
import numpy as np
import json

with open("/Users/daewon/desktop/deep-i/python/PVC파이프/poly.json", "r") as f:
    json_data = json.load(f)

# print(json_data)
# print(json_data["coordinate"][0]["x"])

# cv2
img_path = "/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_152428.jpg"
clicked_points = []
ird = cv2.imread(img_path)
reset = ird.copy();
points = []
count = 0;

poly = []
for data in json_data['coordinate']:
    poly.append([int(data['x']), int(data['y'])])

print()


# 원본 파일을 가져 와서 clicked_points에 있는 점들을 그린다. , point[0] = x, point[1] = y
for point in poly:
    count += 1
    cv2.circle(ird, (point[0], point[1]), 2, (0, 255, 0), thickness=20)

poly_np = np.array(poly, dtype = np.int32)

# points = np.array(clicked_points, dtype=np.int32)
# print(points[count-1])
cv2.imshow("image", ird)

# fpoints = np.array(clicked_points, dtype=np.int32)
key = cv2.waitKey()
if key == 13:
    cv2.polylines(ird, [poly_np], True, (0,255,0), 4)
    #cv2.fillPoly(ird, [fpoints], (0,255,0))
    cv2.imshow("image", ird)

cv2.imshow("image", ird)
cv2.namedWindow("image")
# cv2.setMouseCallback("image", MouseLeftClick)

cv2.waitKey(0)
cv2.destroyAllWindows()
for data in json_data['coordinate']:
    poly.append([int(data['x']), int(data['y'])])
for data in json_data['coordinate']:
print(json_data['coordinate'])
print(json_list)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
runcell(0, '/Users/daewon/Desktop/jsontest.py')
print(json_data["annotations"])
print(json_data["annotations"][0])
print(json_data["annotations"][1])
print(json_data["annotations"][0])
print(json_data["annotations"][0][1])
print(json_data["annotations"]["polygon"][0])
print(json_data["annotations"]["polygon"])
print(json_data["annotations"])
print(json_data["annotations"]['polygon'])
print(json_data["annotations"][0])
print(json_data["annotations"][1])
print(polygon_data["polygon"])
polygon_data = json_data["annotations"]
pd = []
for bu1 in polygon_data:
    pd.append(polygon_data);
print(pd["polygon"])
print(pd[0])
print(pd['polygon'])
print(bu1["polygon"])
print(pd)
print(bu1["polygon"])
print(bu1["polygon"][0])
print(pd)
print(bu1)
for bu1 in polygon_data:
    print(bu1)
for bu1 in polygon_data:
    print(bu1["polygon"])
print(pd)
print(bu1["polygon"][::2])
xpoint = bu1["polygon"][::2]
ypoint = bu1["polygon"][1::2]
xpoint = pd["polygon"][::2]
ypoint = pd["polygon"][1::2]
xmax = []
for i in xpoint:
    if xmax < i:
        xmax = i;

print(xmax)
xmax = []
for i in xpoint:
    if int(xmax) < i:
        xmax = i;

print(xmax)
xmax = xpoint[0]
for i in xpoint:
    if xmax < i:
        xmax = i;

print(xmax)
def MinMax(load_json_data):
    polygon_data = load_json_data["annotations"]
    pd = []
    for bu1 in polygon_data:
        pd.append((bu1["polygon"]))
    
    xpoint = pd["polygon"][::2]
    ypoint = pd["polygon"][1::2]
    
    xmax = xpoint[0]
    for i in xpoint:
        if xmax < i:
            xmax = i
    
    ymax = ypoint[0]
    for j in ypoint:
        if ymax < j:
            ymax = j
    
    xmin = xpoint[0]
    for i in xpoint:
        if xmin > i:
            xmin = i
    
    ymin = ypoint[0]
    for j in ypoint:
        if ymin < j:
            ymin = j
print(ymin)
ymin = ypoint[0]
for j in ypoint:
    if ymin < j:
        ymin = j
print(ymin)
print(ypoint)
print(ymin)
ymin = 0
for j in ypoint:
    if ymin < j:
        ymin = j
print(ymin)
ymin = 0
for k in ypoint:
    if ymin < k:
        ymin = k
print(ymin)
ymin = 0
for j in ypoint:
    if ymin > j:
        ymin = j
print(ymin)
ymin = ypoint[0]
for j in ypoint:
    if ymin > j:
        ymin = j
print(ymin)
ymax = ypoint[0]
for j in ypoint:
    if ymax < j:
        ymax = j

xmin = xpoint[0]
for i in xpoint:
    if xmin > i:
        xmin = i

ymin = ypoint[0]
for j in ypoint:
    if ymin > j:
        ymin = j
print(ymin)
print(ymax)
print(xmin)
print(xmax)
print(ymin)
print(ymax)
print(load_json(json_data))
def load_json(json_data):
    polygon_data = json_data["annotations"]
    pd = []
    for bu1 in polygon_data:
        pd.append((bu1["polygon"]))
    
    return pd

print(load_json(json_data))
print(xpoint)
def yMin(ypoint):
    ymin = ypoint[0]
    for i in ypoint:
        if ymin > i:
            ymin = i
    return ymin
def xMax(xpoint):
    xmax = xpoint[0]
    for i in xpoint:
        if xmax < i:
            xmax = i
    return xmax
    print(xmax)

def yMAx(ypoint):
    ymax = ypoint[0]
    for i in ypoint:
        if ymax < i:
            ymax = i
    return ymax
    print(ymax)

def xMin(xpoint):
    xmin = xpoint[0]
    for i in xpoint:
        if xmin > i:
            xmin = i
    return xmin
    print(xmin)

def yMin(ypoint):
    ymin = ypoint[0]
    for i in ypoint:
        if ymin > i:
            ymin = i
    return ymin
    print(ymin)
xpoint = pd["polygon"][::2]
ypoint = pd["polygon"][1::2]

def xMax(xpoint):
    xmax = xpoint[0]
    for i in xpoint:
        if xmax < i:
            xmax = i
    return xmax
    print(xmax)

def yMAx(ypoint):
    ymax = ypoint[0]
    for i in ypoint:
        if ymax < i:
            ymax = i
    return ymax
    print(ymax)

def xMin(xpoint):
    xmin = xpoint[0]
    for i in xpoint:
        if xmin > i:
            xmin = i
    return xmin
    print(xmin)

def yMin(ypoint):
    ymin = ypoint[0]
    for i in ypoint:
        if ymin > i:
            ymin = i
    return ymin
    print(ymin)
print(yMin(ypoint))
img = cv2.rectangle(ird,((xMin(xpoint),yMin(ypoint)),(xMax(xpoint),yMax(ypoint)),(255,255,0))
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
# points = np.array(clicked_points, dtype=np.int32)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
print(xpoint)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
xpoint = bu1["polygon"][::2]
ypoint = pd["polygon"][1::2]
print(xpoint)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
poly = []
for data in bu1["polygon"]:
    poly.append([int(data['x']), int(data['y'])])
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
for data in bu1['polygon']:
    poly.append([int(data[0]), int(data[1])])

print(poly)
print(bu1["polygon"])
poly = []
for data in bu1:
    poly.append([int(data[0]), int(data[1])])

print(data)
for data in bu1:
    poly.append([int(data["polygon"][0]), int(data["polygon"][1])])

print(data)
print(poly)
poly = []
for data in polygon_data:
    poly.append([int(data["polygon"][0]), int(data["polygon"][1])])

print(poly)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
poly = []
for data in polygon_data:
    poly.append(polygon_data["polygon"])
print(poly)
poly = []
for data in polygon_data:
    poly.append(data["polygon"])
print(poly)
poly = []
for data in polygon_data :
    poly.append(data["polygon"])
print(poly)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
print(poly)
poly = []
for data in polygon_data:
    poly.append(data["polygon"])

print(poly)
print(poly)

poly = []
for data in polygon_data:
    poly.append(data)

print(poly)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
print(point)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
print(polygon_data)
polygon_data = json_data["annotations"]
print(polygon_data)
    json_data = json.load(f)

polygon_data = json_data["annotations"]
print(polygon_data)
polygon_data = json_data["annotations"]
print(polygon_data)
load_json_data = json_data
polygon_data = json_data["annotations"]
print(polygon_data)
polygon_data = load_json_data["annotations"]
print(polygon_data)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
print(pd)
print(polygon_data)
pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))
print(pd)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
with open("/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.json", "r") as f:
    json_data = json.load(f)

load_json_data = json_data
polygon_data = load_json_data["annotations"]
print(polygon_data)

pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))
print(pd)
print(pd)
pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))
    print(pd)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
print("pd" + 1)
print("pd" + str(pd))
result,id = [],0
for i in polygon_data:
    old = id+2
    id = i.rfind('}, {') + 1
    if id > 0:
        if old == id+2:
            if i.rfind('} ]') >= 0: result.append(''.join(['[ ',i[old:]]))
        else: result.append(''.join(['[ ',i[old:id],' ]']))
    else: result.append('X')

for i in result: print(i)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
print(polygon_data)
print(type(polygon_data))
result,id = [],0
for i in polygon_data:
    old = id+2
    id = i.rfind('}, {') + 1
    if id > 0:
        if old == id+2:
            if i.rfind('} ]') >= 0: result.append(''.join(['[ ',i[old:]]))
        else: result.append(''.join(['[ ',i[old:id],' ]']))
    else: result.append('X')

for i in result: print(i)
print(type(id))
print(type(i))
print(type(polygon_data))
result,id = [],0
for i in polygon_data:
    li = list(i)
    old = id + 2
    id = li.rfind('}, {') + 1
    if id > 0:
        if old == id+2:
            if i.rfind('} ]') >= 0: result.append(''.join(['[ ',i[old:]]))
        else: result.append(''.join(['[ ',li[old:id],' ]']))
    else: result.append('X')

for i in result: print(i)

## ---(Fri Aug 19 11:12:33 2022)---
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
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
i=0-
i=0
pd[i]
i+=1
for i in range(len(pd)):
    pd[i]
pd[i]
for j in range(10):
    print(j)
jj = []
for j in range(10):
    print(j)
    jj.append(j)
jj

jj = []
for j in range(10):
    print(j)
    if j %2 == 0:
        jj.append(j)
jjh
jj
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
jj = []
for j in range(10):
    print(j)
    if j %2 == 0:
        jj.append(j)
bbox = []
for bucket in pd:
    bbox.append(xpoint = pd[::2])
for bucket in pd:
    bbox.append(xpoint = pd[::2])
pd = []
pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))
with open("/Users/daewon/desktop/deep-i/python/PVC파이프/20220607_151257.json", "r") as f:
    json_data = json.load(f)

load_json_data = json_data
polygon_data = load_json_data["annotations"]

pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))
pd = []
for bu1 in polygon_data:
    pd.append((bu1["polygon"]))
polygon_data = load_json_data["annotations"]
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
for bucket in pd:
    bbox.append(xpoint = pd[::2])
bbox = []
for bucket in pd:
    bbox.append(xpoint = pd[::2])
bbox = []
for bucket in pd:
    bbox.append(pd[::2])
xbox = []
ybox = []
for bucket in pd:
    xbox.append(bucket[::2])
    ybox.append(bucket[1::2])
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

jj = []
for j in range(10):
    print(j)
    if j %2 == 0:
        jj.append(j)

#bbox = []
#for i in range(len(pd)):
#    val = pd -> bbox
#    bbox.append(val)
#    val로 rectangle

xpoint = bu1["polygon"][::2]
ypoint = bu1["polygon"][1::2]

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
for bucket in pd:
    xbox.append(bucket[::2])
    ybox.append(bucket[1::2])
    img = cv2.rectangle(ird, (xMin(xpoint), yMin(ypoint)), (xMax(xpoint), yMax(ypoint)), (255, 0, 0), 7)
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
xpoint = xbox.append(bucket[::2])
ypoint = ybox.append(bucket[1::2])
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
xpoint = []
ypoint = []
for bucket in pd:
    xpoint = xbox.append(bucket[::2])
    ypoint = ybox.append(bucket[1::2])
xpoint = []
ypoint = []
for bucket in pd:
    xbox.append(bucket)
    xpoint = xbox[::2]
xbox = []
ybox = []
xpoint = []
ypoint = []
for bucket in pd:
    xbox.append(bucket)
    xpoint = xbox[::2]
    ybox.append(bucket)
    ypoint = ybox[1::2]
runcell(0, '/Users/daewon/.spyder-py3/temp.py')
xbox = []
ybox = []
xpoint = []
ypoint = []
for bucket in pd:
    xbox.append(bucket)
    xpoint.append(xbox[::2])
    ybox.append(bucket)
    ypoint.append(ybox[1::2])
for bucket in pd:
    xbox.append(bucket)
    print(xbox)
    # xpoint.append(xbox[::2])
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