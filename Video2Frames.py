"""
将视频转化为图片序列
创作日：2024年03月25日
"""
import os, os.path
import cv2 as cv

video_path = 'F:/Experiment Data/9 DOD ON CHIP/211022/3/Export_20211022_195533/'
cap = cv.VideoCapture(video_path+'20240310_scaleBar_video.avi')
# 定义图片保存路径和帧数计数器
frame_count = 0
Target_folder = video_path+'toframe/'
# 检查是否成功打开视频文件
if not cap.isOpened():
    print("Error opening video file.")
else:
    # 循环读取每一帧直到没有帧可读
    while cap.isOpened():
        print(frame_count)
        ret, frame = cap.read()
        # 如果读取不成功（视频结束），则退出循环
        if not ret:
            break
        # 将当前帧保存为图片
        filename = f"frame_{frame_count}.jpg"
        filepath = os.path.join(Target_folder, filename)
        cv.imwrite(filepath, frame)

        # 帧数计数器加1
        frame_count += 1

    # 关闭视频文件
    cap.release()

print(f"Successfully extracted {frame_count} frames from the video.")

# 确保目录存在
if not os.path.exists(Target_folder):
    os.makedirs(Target_folder)

