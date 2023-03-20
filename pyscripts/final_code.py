# 基于FLANN的匹配器(FLANN based Matcher)定位图片
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import pid
import serial

ser = serial.Serial('com5', 460800, timeout=None)  # 必须用com1这种格式，57600bps，超时:无
print(ser)  # 打印串口的各种设置状态
print(ser.isOpen())  # 检查串口是否打开


ctl=pid.PID(5,0,0)

#plt.ion()
cap = cv2.VideoCapture(1)

cap.set(3,480)
cap.set(4,360)
cap.set(cv2.CAP_PROP_FPS,30)

MIN_MATCH_COUNT =  0 # 设置最低特征点匹配数量为10
blur = 1  # 设置高斯滤波参数
matchesMask = None
# Initiate SIFT detector创建sift检测器
sift = cv2.xfeatures2d.SIFT_create()

# 创建设置FLANN匹配
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

for i in range(0,20):
    ret, old_frame = cap.read()
plt.imshow(old_frame, 'gray')
plt.show()
# 进行高斯滤波模糊
old_frame = cv2.GaussianBlur(old_frame, (blur, blur), 0)

# 找到原始灰度图
template = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  # queryImage

kp1, des1 = sift.detectAndCompute(template, None)

summary_x, summary_y = 0, 0
delta, time1 = 0, time.time()

while 1:
    ret, old_frame = cap.read()
    # 进行高斯滤波模糊
    old_frame = cv2.GaussianBlur(old_frame, (blur, blur), 0)
    # 找到原始灰度图
    target = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  # trainImage

    # find the key points and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(target, None)

    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    # 舍弃大于0.7的匹配
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        # 获取关键点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        nums = src_pts.shape[0]
        src_pts.shape = (nums, 2)
        dst_pts.shape = (nums, 2)
        [shift_x, shift_y] = np.hsplit(dst_pts - src_pts, 2)

        mean = np.mean(shift_y)
        var = np.var(shift_y)
        # print("原始数据共", len(shift_y), "个")
        # print("中位数:",np.median(deg))
        percentile = np.percentile(shift_y, (25, 50, 75), interpolation='midpoint')
        # 以下为箱线图的五个特征值
        Q1 = percentile[0]  # 上四分位数
        Q3 = percentile[2]  # 下四分位数
        IQR = Q3 - Q1  # 四分位距
        ulim = Q3 + 1.5 * IQR  # 上限 非异常范围内的最大值
        llim = Q1 - 1.5 * IQR  # 下限 非异常范围内的最小值

        new_y = []
        for i in range(len(shift_y)):
            if llim < shift_y[i] < ulim:
                new_y.append(shift_y[i])
        # print("清洗后数据共", len(new_y), "个\n")

        delta = (int)((time.time() - time1)*1000)
        time1 = time.time()
        a = int(np.mean(new_y))
        print('%.2f'%a, '\t', delta,'\t',len(new_y),ctl.output)
        ctl.update(a)
        ser.write(str(int(ctl.output)).encode())

        while ser.inWaiting() == 0  :
              pass
        ser.flushInput()
        '''
        # 计算变换矩阵和MASK
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = template.shape
        # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        cv2.polylines(target, [np.int32(dst)], True, 0, 2, cv2.LINE_AA)
        '''
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
    '''   
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)
    result = cv2.drawMatches(template, kp1, target, kp2, good, None, **draw_params)
    plt.imshow(result, 'gray')
    plt.show()
    plt.pause(0.00000000001)
    plt.close('gray')
    '''
    # https://blog.csdn.net/zhuisui_woxin/article/details/84400439
