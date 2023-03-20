import numpy as np
import cv2
import time

cap = cv2.VideoCapture(1)


# ShiTomasi 角点检测参数
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7)

# lucas kanade光流法参数
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 创建随机颜色

color = np.random.randint(0,255,(100,3))

# 获取第一帧，找到角点
ret, old_frame = cap.read()
#找到原始灰度图
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

#获取图像中的角点，返回到p0中
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# 创建一个蒙版用来画轨迹
mask = np.zeros_like(old_frame)

imgInfo = old_frame.shape
height= imgInfo[0]
width = imgInfo[1]

time1 = time.time()
a1,b1,c1,d1 = 0,0,0,0
position_x,position_y = width/2,height/2

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(p0)
    # 计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # 选取好的跟踪点
    good_new = p1[st==[1]]
    good_old = p0[st==[1]]

    # 画出轨迹
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        a1 += a
        b1 += b
        c1 += c
        d1 += d
    a1 /= i+1
    b1 /= i+1
    c1 /= i+1
    d1 /= i+1
    delta = time.time()-time1
    time1 = time.time()
    position_x -= (c-a)
    position_y -= (d-b)
    #mask = cv2.line(mask,(int(a1),int(b1)),(int(c1),int(d1)),color[i].tolist(),2)
    print(i+1)
    #frame = cv2.circle(frame,(int(a1),int(b1)),5,color[i].tolist(),-1)
    frame = cv2.circle(frame,(int(position_x),int(position_y)),5,color[i].tolist(),-1)
    #print(a,b)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
        break

    # 更新上一帧的图像和追踪点
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
