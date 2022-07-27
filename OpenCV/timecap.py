import cv2
import schedule
import time
import datetime

cam = cv2.VideoCapture(1)
fps = cam.get(cv2.CAP_PROP_FPS)

cv2.namedWindow("test")

img_counter = 0
ret, frame = cam.read()
now = datetime.datetime.now()

def job():
    global img_counter, ret, frame, now
    
    img_name = "result/opencv_frame_{}.png".format(img_counter)
    cv2.imwrite(img_name, frame)
    print("{} written!".format(img_name))
    img_counter += 1
    
    
job()
schedule.every(60).seconds.do(job)
    
while (cam.isOpened()):
    ret, frame = cam.read()
    try:
        cv2.imshow("test", frame)
    except Exception as err:
            print(err)
    
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing…")
        break
    
    schedule.run_pending()
    time.sleep(1)    

cam.release()
cv2.destroyAllWindows()