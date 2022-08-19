# 원본 파일을 가져 와서 clicked_points에 있는 점들을 그린다. , point[0] = x, point[1] = y
for point in bu1["polygon"]:
    cv2.circle(ird, (point[0], point[1]), 2, (0, 255, 0), thickness=20)

poly_np = np.array(point, dtype = np.int32)
cv2.fillPoly(ird, [poly_np], (0,255,0))
cv2.imshow("image", ird)

img = cv2.rectangle(ird, (xMin(xpoint), yMin(ypoint)), (xMax(xpoint), yMax(ypoint)), (255,0, 0))



jj = []
for j in range(10):
    print(j)
    if j %2 == 0:
        jj.append(j)