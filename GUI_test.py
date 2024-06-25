import cv2
import numpy as np
import tkinter as tk
from tkinter import Variable, filedialog
from PIL import Image, ImageTk

def nothing(x):
    pass

# # 定義全域變數
# drawing = False # 是否正在畫矩形
# ix, iy = -1, -1 # 起始點坐標
# x1, y1, x2, y2 = -1, -1, -1, -1 # 矩形範圍坐標

# # 滑鼠回調函數
# def draw_rectangle(event, x, y, flags, param):
#     global ix, iy, x1, y1, x2, y2, drawing

#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy = x, y
#         x1, y1 = x, y

#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing:
#             x2, y2 = x, y

#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         x2, y2 = x, y

# # 創建一個 Tkinter 主窗口
# root = tk.Tk()
# root.withdraw()  # 隱藏主窗口

# # 打開文件選擇對話框
# file_path = filedialog.askopenfilename(title="選擇圖片",
#                                        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])

# # 檢查是否選擇了文件
# if file_path:
#     # 使用原始字符串處理文件路徑
#     file_path = r"{}".format(file_path)

#     # 讀取選擇的圖片
#     image = cv2.imread(file_path)

#     # 檢查影像是否正確讀取
#     if image is None:
#         print("Error: Unable to read image")
#     else:
#         # 將圖像轉換為 HSV 色彩空間
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#         # 創建一個窗口
#         cv2.namedWindow('Trackbars')

#         # 創建6個trackbars來調整HSV範圍
#         cv2.createTrackbar('Lower Hue', 'Trackbars', 19, 179, nothing)
#         cv2.createTrackbar('Lower Saturation', 'Trackbars', 56, 255, nothing)
#         cv2.createTrackbar('Lower Value', 'Trackbars', 35, 255, nothing)
#         cv2.createTrackbar('Upper Hue', 'Trackbars', 50, 179, nothing)
#         cv2.createTrackbar('Upper Saturation', 'Trackbars', 255, 255, nothing)
#         cv2.createTrackbar('Upper Value', 'Trackbars', 255, 255, nothing)

#         while True:
#             # 獲取trackbars的當前位置
#             lower_hue = cv2.getTrackbarPos('Lower Hue', 'Trackbars')
#             lower_saturation = cv2.getTrackbarPos('Lower Saturation', 'Trackbars')
#             lower_value = cv2.getTrackbarPos('Lower Value', 'Trackbars')
#             upper_hue = cv2.getTrackbarPos('Upper Hue', 'Trackbars')
#             upper_saturation = cv2.getTrackbarPos('Upper Saturation', 'Trackbars')
#             upper_value = cv2.getTrackbarPos('Upper Value', 'Trackbars')

#             # 定義黃色的 HSV 範圍
#             lower_yellow = np.array([lower_hue, lower_saturation, lower_value])
#             upper_yellow = np.array([upper_hue, upper_saturation, upper_value])

#             # 創建遮罩
#             mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

#             # 將遮罩應用於原圖像
#             yellow_only = cv2.bitwise_and(image, image, mask=mask)

#             # 合併圖像
#             combined_image = np.hstack((image, yellow_only))
#             # 顯示合併圖像
#             cv2.imshow('Combined Image', combined_image)

#             # 等待按鍵按下，按 'q' 鍵退出
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         # 關閉所有窗口
#         cv2.destroyAllWindows()

#         # 在此加入矩形選擇範圍的代碼
#         clone = yellow_only.copy()

#         cv2.namedWindow('image')
#         cv2.setMouseCallback('image', draw_rectangle)

#         while True:
#             img = clone.copy()
#             if drawing:
#                 cv2.rectangle(img, (ix, iy), (x2, y2), (0, 255, 0), 2)
#             #print(ix,iy,x2,y2,x1,y1)
            
#             cv2.imshow('image', img)
#             key = cv2.waitKey(1) & 0xFF

#             if key == ord('r'):
#                 clone = combined_image.copy()
#                 x1, y1, x2, y2 = -1, -1, -1, -1

#             if key == ord('c'):
#                 if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
#                     # 確保 x1, y1 是左上角，x2, y2 是右下角
#                     x1, x2 = min(x1, x2), max(x1, x2)
#                     y1, y2 = min(y1, y2), max(y1, y2)
                    
#                     # 使用 numpy 方式裁切圖片
#                     cropped_image = image[y1:y2, x1:x2]
#                     cropped_h, cropped_w = cropped_image.shape[:2]
                    
#                     # 打印裁切後的尺寸
#                     print((x2-x1), (y2-y1))
#                     print(cropped_w, cropped_h)

#                     # 設定新的尺寸，這裡是放大4倍
#                     new_size = (cropped_w*4, cropped_h*4)
                    
#                     # 放大影像
#                     resized_cropped_image = cv2.resize(cropped_image, new_size, interpolation=cv2.INTER_LINEAR)
                    
#                     # 顯示裁切後的圖像
#                     cv2.imshow('Result*4', resized_cropped_image)
                    
#                     cv2.waitKey()
#                     cv2.destroyAllWindows()
#                     break
# else:
#     print("No file selected")

#################################################
#換成使用tkinter製作GUI使用opencv
#但是上面讀取檔案一次照片過後會無法在讀取
#################################################



# 初始化Tkinter視窗
window = tk.Tk()
window.title("tkinter製作GUI使用opencv")
window.geometry("1200x700+50+50")  #從螢幕上的(50, 50)開啟
window.resizable(width=False, height=False)
window.configure(bg="white")  # 設置背景色為白色

# 創建4個Frame
img_frame = tk.Frame(window,bg="yellow", width=300, height=700)
img_frame.grid_propagate(False)  # 防止 frame 自動調整大小
canvas = tk.Canvas(img_frame, width=300, height=700, bg='#fff')
canvas.pack(fill="both", expand=True)

frame1 = tk.Frame(window,bg="red", width=300, height=700)
frame1.grid_propagate(False)  # 防止 frame 自動調整大小

frame2 = tk.Frame(window,bg="green", width=300, height=700)
frame2.grid_propagate(False)  # 防止 frame 自動調整大小

frame3 = tk.Frame(window,bg="blue", width=300, height=700)
frame3.grid_propagate(False)  # 防止 frame 自動調整大小

# 將4個Frame放置在主窗口中，並進行網格布局
img_frame.grid(row=0, column=0, rowspan=3)# 圖像 Frame 占據左側的多行
frame1.grid(row=0, column=1, padx=10)
frame2.grid(row=0, column=2, padx=10)
frame3.grid(row=0, column=3, padx=10)
# 全局變數
image = None
image_label = None
imgtk = None
def open_image():
    global image, image_label
    file_path = filedialog.askopenfilename()
   
    if file_path:
        image = cv2.imread(file_path)
        display_image(image)
        original_photo()

def display_image(image):
    global image_label, imgtk
    bgr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #轉換圖像顏色從BGR到RGB
    im_pil = Image.fromarray(bgr_image)  #將圖像轉換為PIL格式
    imgtk = ImageTk.PhotoImage(image=im_pil)  #將PIL圖像轉換為Tkinter可用的格式

    if image_label is None:
        image_label = tk.Label(frame1, image=imgtk)
        image_label.image = imgtk# 保存引用
        image_label.grid(row=0, column=0)
    else:
        image_label.configure(image=imgtk)
        image_label.image = imgtk  # 更新引用

def original_photo():
    canvas.delete('all')  # 清空 Canvas 原本內容
    canvas.create_image(0, 0, anchor='nw', image=imgtk)  # 建立圖片
    canvas.imgtk = imgtk  # 更新引用
    display_image(cv2.cvtColor(image))

def convert_to_grayscale():
    global image
    if image is not None:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        display_image(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB))

def edge_detection():
    global image
    if image is not None:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)
        display_image(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

def update_hsv_filter(val):
    if image is not None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_hue = scale_lower_hue.get()
        lower_saturation = scale_lower_saturation.get()
        lower_value = scale_lower_value.get()
        upper_hue = scale_upper_hue.get()
        upper_saturation = scale_upper_saturation.get()
        upper_value = scale_upper_value.get()

        lower_bound = (lower_hue, lower_saturation, lower_value)
        upper_bound = (upper_hue, upper_saturation, upper_value)

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        result = cv2.bitwise_and(image, image, mask=mask)
        display_image(result)


def nothing(val):
    pass

def create_hsv_trackbars():
    global scale_lower_hue, scale_lower_saturation, scale_lower_value
    global scale_upper_hue, scale_upper_saturation, scale_upper_value

    tk.Label(frame3, text="Lower Hue").grid(row=0, column=1)
    scale_lower_hue = tk.Scale(frame3, from_=0, to=179, orient=tk.HORIZONTAL, command=update_hsv_filter)
    scale_lower_hue.set(19)
    scale_lower_hue.grid(row=0, column=2)

    tk.Label(frame3, text="Lower Saturation").grid(row=1, column=1)
    scale_lower_saturation = tk.Scale(frame3, from_=0, to=255, orient=tk.HORIZONTAL, command=update_hsv_filter)
    scale_lower_saturation.set(56)
    scale_lower_saturation.grid(row=1, column=2)

    tk.Label(frame3, text="Lower Value").grid(row=2, column=1)
    scale_lower_value = tk.Scale(frame3, from_=0, to=255, orient=tk.HORIZONTAL, command=update_hsv_filter)
    scale_lower_value.set(35)
    scale_lower_value.grid(row=2, column=2)

    tk.Label(frame3, text="Upper Hue").grid(row=3, column=1)
    scale_upper_hue = tk.Scale(frame3, from_=0, to=179, orient=tk.HORIZONTAL, command=update_hsv_filter)
    scale_upper_hue.set(50)
    scale_upper_hue.grid(row=3, column=2)

    tk.Label(frame3, text="Upper Saturation").grid(row=4, column=1)
    scale_upper_saturation = tk.Scale(frame3, from_=0, to=255, orient=tk.HORIZONTAL, command=update_hsv_filter)
    scale_upper_saturation.set(255)
    scale_upper_saturation.grid(row=4, column=2)

    tk.Label(frame3, text="Upper Value").grid(row=5, column=1)
    scale_upper_value = tk.Scale(frame3, from_=0, to=255, orient=tk.HORIZONTAL, command=update_hsv_filter)
    scale_upper_value.set(255)
    scale_upper_value.grid(row=5, column=2)

    # 將此函數綁定到每個 Scale 組件上
    scale_lower_hue.bind('<MouseWheel>', on_scale_mousewheel)
    scale_lower_saturation.bind('<MouseWheel>', on_scale_mousewheel)
    scale_lower_value.bind('<MouseWheel>', on_scale_mousewheel)
    scale_upper_hue.bind('<MouseWheel>', on_scale_mousewheel)
    scale_upper_saturation.bind('<MouseWheel>', on_scale_mousewheel)
    scale_upper_value.bind('<MouseWheel>', on_scale_mousewheel)
#滑鼠滾輪來調整scale
def on_scale_mousewheel(event):
    scale = event.widget
    delta = -1 if event.delta < 0 else 1  # 滾輪向下滾動時 delta 為負數，向上滾動時 delta 為正數
    scale.set(scale.get() + delta)
     

# 定義選擇區域的回呼函數
def select_area(event, x, y, flags, param):
    global refPt, cropping, image

    # 當按下左鍵時記錄初始點
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # 當左鍵鬆開時記錄結束點，並畫出矩形
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        # 在圖像上畫出矩形
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

# 匯入圖像的函數
def import_image():
    global image, refPt, cropping, canvas

    # 彈出文件對話框選擇圖像文件
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if filepath:
        image = cv2.imread(filepath)  # 讀取圖像
        clone = image.copy()  # 複製一份圖像備份
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", select_area)  # 設置滑鼠回呼函數
        
        while True:
            cv2.imshow("image", image)  # 顯示圖像
            key = cv2.waitKey(1) & 0xFF  # 等待鍵盤輸入
            if key == ord("r"):  # 如果按下 'r' 鍵，重置圖像
                image = clone.copy()
            elif key == ord("c"):  # 如果按下 'c' 鍵，退出循環
                break

        if len(refPt) == 2:  # 如果選擇了兩個點
            roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]  # 提取選擇的區域
            W, H = roi.shape[:2]
            new_size = (W*4, H*4) 
            resize_roi = cv2.resize(roi, new_size, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("ROI", resize_roi)  # 顯示選擇的區域
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()  # 關閉所有窗口

        # 將處理過的圖像顯示在Tkinter的Canvas中
        show_image_on_canvas(resize_roi)

# 將處理過的圖像顯示在Canvas上的函數
def show_image_on_canvas(img):
    global canvas_img
    cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換圖像顏色從BGR到RGB
    img_pil = Image.fromarray(cv2image)  # 將圖像轉換為PIL格式
    imgtk = ImageTk.PhotoImage(image=img_pil)  # 將PIL圖像轉換為Tkinter可用的格式
    
    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)  # 在Canvas上顯示圖像
    canvas.image = imgtk  # 保持圖像的引用以避免被垃圾回收
    canvas_img = img_pil  # 保存圖像以便儲存

# 儲存圖像的函數
def save_image():
    if canvas_img:
        # 彈出文件對話框選擇保存位置
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("BMP files", "*.bmp")])
        if filepath:
            canvas_img.save(filepath)  # 保存圖像

#定義結束應用程式的函數
def exit_application(event):
    window.destroy()

# 綁定 'Q' 鍵事件到 exit_application 函數
window.bind('<q>', exit_application)

# 創建按鈕並綁定功能
btn_open = tk.Button(frame2
                     ,text="Open Image" 
                     , command=open_image   
                     ,font=('Arial',12,'bold')   
                     ,padx=5   
                     ,pady=5
                     ,activeforeground='#f00')
btn_open.pack(fill='x', expand=1)

btn_gray = tk.Button(frame2 
                     , text="Convert to Grayscale"
                     , command=convert_to_grayscale
                     ,font=('Arial',12,'bold')   
                     ,padx=5   
                     ,pady=5
                     ,activeforeground='#f00')
btn_gray.pack(fill='x', expand=1)

btn_edge = tk.Button(frame2 
                     , text="Edge Detection"
                     , command=edge_detection
                     ,font=('Arial',12,'bold')   
                     ,padx=5   
                     ,pady=5
                     ,activeforeground='#f00')
btn_edge.pack(fill='x', expand=1)

btn_hsv = tk.Button(frame2  
                    , text="Create HSV Trackbars去背"
                    , command=create_hsv_trackbars
                    ,font=('Arial',12,'bold')   
                    ,padx=5   
                    ,pady=5
                    ,activeforeground='#f00')
btn_hsv.pack(fill='x', expand=1)
# 選取花蕊cut and resize*4
btn_cut = tk.Button(frame2  
                    , text="cut and resize*4"
                    , command=import_image
                    ,font=('Arial',12,'bold')   
                    ,padx=5   
                    ,pady=5
                    ,activeforeground='#f00')
btn_cut.pack(fill='x', expand=1)
# 添加儲存照片按鈕
btn_save = tk.Button(frame2 
                    , text="save"
                    , command=save_image
                    ,font=('Arial',12,'bold')   
                    ,padx=5   
                    ,pady=5
                    ,activeforeground='#f00')
btn_save.pack(fill='x', expand=1)
# 開始Tkinter主迴圈
window.mainloop()
