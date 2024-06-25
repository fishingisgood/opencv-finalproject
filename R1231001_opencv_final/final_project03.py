import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import datetime
import subprocess
def nothing(x):
    pass

# 定義全域變數
drawing = False # 是否正在畫矩形
ix, iy = -1, -1 # 起始點坐標
x1, y1, x2, y2 = -1, -1, -1, -1 # 矩形範圍坐標

# 滑鼠回調函數
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, x1, y1, x2, y2, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        x1, y1 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            x2, y2 = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y


# 創建一個 Tkinter 主窗口
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

    # 檢查影像是否正確讀取
    if image is None:
        print("Error: Unable to read image")
    else:
        # 將圖像轉換為 HSV 色彩空間
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 創建一個窗口
        cv2.namedWindow('Trackbars')

        # 創建6個trackbars來調整HSV範圍
        cv2.createTrackbar('Lower Hue', 'Trackbars', 19, 179, nothing)
        cv2.createTrackbar('Lower Saturation', 'Trackbars', 56, 255, nothing)
        cv2.createTrackbar('Lower Value', 'Trackbars', 35, 255, nothing)
        cv2.createTrackbar('Upper Hue', 'Trackbars', 50, 179, nothing)
        cv2.createTrackbar('Upper Saturation', 'Trackbars', 255, 255, nothing)
        cv2.createTrackbar('Upper Value', 'Trackbars', 255, 255, nothing)

        while True:
            # 獲取trackbars的當前位置
            lower_hue = cv2.getTrackbarPos('Lower Hue', 'Trackbars')
            lower_saturation = cv2.getTrackbarPos('Lower Saturation', 'Trackbars')
            lower_value = cv2.getTrackbarPos('Lower Value', 'Trackbars')
            upper_hue = cv2.getTrackbarPos('Upper Hue', 'Trackbars')
            upper_saturation = cv2.getTrackbarPos('Upper Saturation', 'Trackbars')
            upper_value = cv2.getTrackbarPos('Upper Value', 'Trackbars')

            # 定義黃色的 HSV 範圍
            lower_yellow = np.array([lower_hue, lower_saturation, lower_value])
            upper_yellow = np.array([upper_hue, upper_saturation, upper_value])

            # 創建遮罩
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # 將遮罩應用於原圖像
            yellow_only = cv2.bitwise_and(image, image, mask=mask)

            # 合併圖像
            combined_image = np.hstack((image, yellow_only))
            # 顯示合併圖像
            cv2.imshow('HSV_remove', combined_image)

            # 等待按鍵按下，按 'q' 鍵退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 關閉所有窗口
        cv2.destroyAllWindows()

        # 在此加入矩形選擇範圍的代碼
        clone = yellow_only.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_rectangle)

        while True:
            img = clone.copy()
            if drawing:
                cv2.rectangle(img, (ix, iy), (x2, y2), (0, 255, 0), 2)
            #print(ix,iy,x2,y2,x1,y1)
            
            cv2.imshow('image', img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                clone = combined_image.copy()
                x1, y1, x2, y2 = -1, -1, -1, -1

            if key == ord('c'):
                if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
                    # 確保 x1, y1 是左上角，x2, y2 是右下角
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    # 使用 numpy 方式裁切圖片
                    cropped_image = image[y1:y2, x1:x2]
                    cropped_h, cropped_w = cropped_image.shape[:2]
                    
                    # 打印裁切後的尺寸
                    print((x2-x1), (y2-y1))
                    print(cropped_w, cropped_h)

                    # 設定新的尺寸，這裡是放大8倍
                    ratio = 8
                    new_size = (cropped_w*ratio, cropped_h*ratio)
                    
                    # 放大影像
                    resized_cropped_image = cv2.resize(cropped_image, new_size, interpolation=cv2.INTER_LINEAR)
                    
                    # 顯示裁切後的圖像
                    cv2.imshow(F'Result*{ratio}',resized_cropped_image)
                    cv2.waitKey()
                    cv2.destroyAllWindows()

                    gray = cv2.cvtColor(resized_cropped_image, cv2.COLOR_BGR2GRAY)
                    # 模糊化
                    gray2 = cv2.medianBlur(gray, 5) 
                    #自適應閾值方法
                    output2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

                    #直方圖均勻化equalizeHist(src)
                    src = gray
                    plt.subplot(221)                        # 建立子圖 1
                    plt.imshow(src, 'gray')                 # 灰階顯示第1張圖
                    plt.subplot(222)                        # 建立子圖 2
                    plt.hist(src.ravel(),256)               # 降維再繪製直方圖
                    plt.subplot(223)                        # 建立子圖 3
                    dst = cv2.equalizeHist(src)             # 均衡化處理
                    plt.imshow(dst, 'gray')                 # 顯示執行均衡化的結果影像
                    plt.subplot(224)                        # 建立子圖 4
                    plt.hist(dst.ravel(),256)               # 降維再繪製直方圖
                    plt.show()
                    #自適應閾值方法
                    output3 = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

                    #邊緣檢測
                    edges = cv2.Canny(gray, 50, 50)
                    

                    gray_adaptiveTreshold_edge = np.hstack((gray,output2,edges))
                    cv2.imshow('medianBlur+adaptiveThreshold+edges', gray_adaptiveTreshold_edge)

                    edges2 = cv2.Canny(output3, 50, 50)
                    gray_equalizeHist_edge = np.hstack((src,dst,output3,edges2))
                    cv2.imshow('equalizeHist+adaptiveThreshold+edges2', gray_equalizeHist_edge)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
                    # 取得當前日期和時間
                    now = datetime.datetime.now()
                    # 格式化日期和時間為字串，並用於生成文件名
                    # 格式例如：20240612_101530
                    F = 'female'
                    M = 'male'
                    filename = F + now.strftime("%Y%m%d_%H%M%S") + ".jpg"
                    print(f"生成的文件名是：{filename}")
                    cv2.imshow('equalizeHist Image', dst)
                    cv2.waitKey(ord('s'))
                    cv2.imwrite(filename, dst)
                    cv2.waitKey
                    cv2.destroyAllWindows()
                    #接下來是小畫家子程式因為TKinter檔案總管輸入照片只能一次# 打開小畫家(F12另存新檔)
                    process = subprocess.Popen(['mspaint'])
                    process.wait()  # 等待腳本執行完成
                    process = subprocess.Popen(['python', r'C:\test\opencv_final\ch23_2_R2.py'])
                    process.wait()  # 等待腳本執行完成
  

                    break
else:
    print("No file selected")