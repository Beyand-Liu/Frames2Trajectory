# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

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
        if os.path.splitext(file)[1] == '.jpg':
            #  os.path.splitext()是一个元组,类似于('188739', '.jpg')，索引1可以获得文件的扩展名
            num = num + 1
    return num

def IMfill(img):
    # Mask
    img_floodfill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    #show(mask)
    # Floodfill from point (0,0)
    cv.floodFill(img_floodfill, mask, (0, 0), 255)
    #show(img_floodfill)
    img_floodfill_inv = cv.bitwise_not(img_floodfill)
    #show(img_floodfill_inv)
    img_dst = img | img_floodfill_inv
    #show(img_dst)
    #img_dst = img_th | img_floodfill
    return img_dst

# ---------------------- MAIN -----------------------#
# ------ ROI ----------#
img_Wid = 256 #528
img_Hei = 1102 #1487
Hei_adj = 1

pixelLen = 8.7 # um/pixel
filepath = 'D:/Master/Experiment Data/ManyBouncingDroplets/'
framesNum = filenum(filepath)
print(framesNum)
bin0 = np.full((img_Hei-Hei_adj, img_Wid), 255, dtype=np.uint8)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))  # 开运算卷积核

for i in range(1, framesNum, 5):    #130
    imgi = cv.imread(filepath + "frame_" + str(i) + '.jpg')             # 注意图片格式
    #img_g = cv.cvtColor(imgi[55:img_Hei, :, :], cv.COLOR_BGR2GRAY)
    img_g = cv.cvtColor(imgi[Hei_adj:img_Hei, :, :], cv.COLOR_BGR2GRAY)           # 150:1102
    bini = cv.adaptiveThreshold(img_g, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 6)
    #show(bini)
    bini = cv.morphologyEx(bini, cv.MORPH_CLOSE, kernel)  # 形态学 去除噪点
    #bini = IMfill(bini)

    bini = ~bini // 255
    #img_dst = cv.bitwise_and(bin0, bini)
    bin0 = cv.bitwise_and(bin0, bin0, mask=bini)
    #show(bin0)

bin0 = cv.morphologyEx(bin0, cv.MORPH_CLOSE, kernel)

bin255=bin0*255
cv.line(bin255, (30,60), (30+int(200/pixelLen), 60), 0, 5, cv.LINE_AA)
cv.putText(bin255, " 200 um", (20,52), cv.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2)
show(bin255)
