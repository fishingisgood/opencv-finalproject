# ch23_2.py
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import os
import datetime
import subprocess
plt.rcParams["font.family"] = ["Microsoft JhengHei"]


root = tk.Tk()
root.withdraw()  # 隱藏主窗口
# 打開文件選擇對話框
file_path = filedialog.askopenfilename(title="選擇圖片",
                                       filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])

# 檢查是否選擇了文件
if file_path:
    # 使用原始字符串處理文件路徑
    file_path = r"{}".format(file_path)

    # 讀取選擇的圖片
    image = cv2.imread(file_path)
#print(file_path)
# 獲取文件名
file_name = os.path.basename(file_path)
# 去掉文件擴展名
file_name = os.path.splitext(file_name)[0]
# 打印結果
print(file_name)


src = cv2.imread(f'{file_name}.jpg')                # 讀取影像
mask = np.zeros(src.shape[:2],np.uint8)     # 建立遮罩, 大小和src相同
bgdModel = np.zeros((1,65),np.float64)      # 建立內部用暫時計算陣列
fgdModel = np.zeros((1,65),np.float64)      # 建立內部用暫時計算陣列
rect = (10,30,380,360)                      # 建立ROI區域
# 呼叫grabCut()進行分割 
cv2.grabCut(src,mask,rect,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_RECT)
maskpict = cv2.imread(f'{file_name}_mask.jpg')      # 讀取影像
newmask = cv2.imread(f'{file_name}_mask.jpg',cv2.IMREAD_GRAYSCALE)  # 灰階讀取
newmask_r = cv2.resize(newmask, (src.shape[1], src.shape[0]))
mask[newmask_r == 0] = 0                      # 白色內容則確定是前景
mask[newmask_r == 255] = 1                    # 黑色內容則確定是背景
cv2.grabCut(src,mask,None,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_MASK)
mask = np.where((mask==0)|(mask==2),0,1).astype('uint8')
dst = src * mask[:,:,np.newaxis]               # 計算輸出影像
src_rgb = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)   # 將BGR轉RGB
maskpict_rgb = cv2.cvtColor(maskpict,cv2.COLOR_BGR2RGB)
dst_rgb = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)   # 將BGR轉RGB
plt.subplot(131)
plt.title("原始影像")
plt.imshow(src_rgb)
plt.axis('off')
plt.subplot(132)
plt.title("遮罩影像")
plt.imshow(maskpict_rgb)
plt.axis('off')
plt.subplot(133)
plt.title("擷取影像")
plt.imshow(dst_rgb)
plt.axis('off')
#plt.show()


src_rgb_maskpict_rgb_dst_rgb = np.hstack((src_rgb,maskpict_rgb,dst_rgb))
cv2.imshow('src_rgb+maskpict_rgb+dst_rgb', src_rgb_maskpict_rgb_dst_rgb)
cv2.imwrite(f'{file_name}R.jpg', dst_rgb)
cv2.waitKey()
cv2.destroyAllWindows()