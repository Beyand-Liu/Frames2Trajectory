"""
将多帧图片合成到一张中显示运动过程和姿态，并画出运动轨迹
创作日：2024年03月24日
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os, os.path


def show(img):
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def cvshow(img):
    cv.imshow('IMAGE', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 查找文件夹中指定文件的数目
def filenum(img_path):
    num = 0
    count_path = img_path
    files = os.listdir(count_path)
    for file in files:
        if os.path.splitext(file)[1] == '.bmp':
            #  os.path.splitext()是一个元组,类似于('188739', '.jpg')，索引1可以获得文件的扩展名
            num = num + 1
    return num


def IMfill(img):
    # Mask
    img_floodfill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # show(mask)
    # Floodfill from point (0,0)
    cv.floodFill(img_floodfill, mask, (0, 0), 255)
    # show(img_floodfill)
    img_floodfill_inv = cv.bitwise_not(img_floodfill)
    # show(img_floodfill_inv)
    img_dst = img | img_floodfill_inv
    # show(img_dst)
    # img_dst = img_th | img_floodfill
    return img_dst

def cnt_area(cnt):
    area = cv.contourArea(cnt)
    return area


# ---------------------- MAIN -----------------------#
# ------ ROI ----------#
img_Wid = 528
img_Hei = 1487
Hei_adj = 60
#----------------------#
pixelLen = 8.7  # um/pixel
filepath = 'D:/Master/Experiment Data/JiayinHeJin1/'
framesNum = filenum(filepath)
print(framesNum)
centroidList=[]

bin0 = np.full((img_Hei - Hei_adj, img_Wid), 255, dtype=np.uint8)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))  # 开运算卷积核

for i in range(3, framesNum-1, 2):
    imgi = cv.imread(filepath + "frame_" + str(i) + '.bmp')  # 注意图片格式
    img_g = cv.cvtColor(imgi[Hei_adj:img_Hei, :, :], cv.COLOR_BGR2GRAY)  # 150:1102
    bini = cv.adaptiveThreshold(img_g, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 6)
    # show(bini)
    bini = cv.morphologyEx(bini, cv.MORPH_CLOSE, kernel)  # 开运算 去除噪点
    bini = IMfill(bini)

    cnts, hiers = cv.findContours(bini, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts=list(cnts)
    cnts.sort(key=cnt_area, reverse=True)  # False 降序排列； True 升序排列
    moments = cv.moments(cnts[0])
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        centroid = (cx, cy)
    else:
        print("Contour has no area.")
        centroid = None
    centroidList.append((cx,cy))
    bini = ~bini // 255
    # bin0 = cv.bitwise_and(bin0, bini)
    bin0 = cv.bitwise_and(bin0, bin0, mask=bini)
    bin0 = cv.morphologyEx(bin0, cv.MORPH_CLOSE, kernel)

imgM = cv.cvtColor(bin0, cv.COLOR_GRAY2RGB)
cx0 = centroidList[0][0]
cy0 = centroidList[0][1]
for j in range(1,len(centroidList)):
    cx = centroidList[j][0]
    cy = centroidList[j][1]
    cv.circle(imgM, (cx0, cy0), radius=4, color=(0, 0, 255), thickness=-1)
    cv.line(imgM, (cx0, cy0), (cx, cy), (255, 0, 0), 2, cv.LINE_AA)
    cx0=cx
    cy0=cy

cv.line(imgM, (30, 60), (30 + int(500 / pixelLen), 60), (0,0,0), 5, cv.LINE_AA)
cv.putText(imgM, " 500 um", (12, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
show(imgM)
