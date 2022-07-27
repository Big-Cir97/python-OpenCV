import cv2
import schedule
import time
import datetime
import requests

cam = cv2.VideoCapture(1)
fps = cam.get(cv2.CAP_PROP_FPS)

cv2.namedWindow("test")

img_counter = 0
ret, frame = cam.read()
now = datetime.datetime.now()

def job():
    global img_counter, ret, frame, now
    
    # 저장될 이미지 경로 / 이름
    img_name = "result/opencv_frame_{}.png".format(img_counter)
    cv2.imwrite(img_name, frame)
    print("{} written!".format(img_name))
    
    # 저장 이미지
    img = '/Users/daewon/.spyder-py3/OpenCV/result/opencv_frame_0.png'
    f = open(img, 'rb')
    files = {"file": (img, f)}

    res = requests.post('http://119.65.127.49:5000/upload',
                    files = files)
    f.close()
    # 응답값 확인
    # code 0 : 전송/받기 실패
    # code 1 : 전송/받기 성공
    # result : 결과 메세지
    res.json();
    
# job 한번 수행 -> 이미지 저장, every() : 시간 설정    
job()
schedule.every(20).seconds.do(job)
    
while (cam.isOpened()):
    ret, frame = cam.read()
    try:
        cv2.imshow("test", frame)
    except Exception as err:
            print(err)
            break
            
            
    schedule.run_pending()
    time.sleep(1)    

cam.release()
cv2.destroyAllWindows()



