import cv2
import numpy as np

# 初始化畫布
canvas = np.ones((500, 500, 3), dtype="uint8") * 255

# 初始筆的顏色和粗細
drawing_color = (0, 0, 0)  # 默認黑色
thickness = 5

# 繪圖狀態
drawing = False

# 繪圖回調函數
def draw(event, x, y, flags, param):
    global drawing, canvas, drawing_color, thickness
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(canvas, (x, y), thickness, drawing_color, -1)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), thickness, drawing_color, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# 創建窗口並設置回調
cv2.namedWindow("Canvas")
cv2.setMouseCallback("Canvas", draw)

while True:
    cv2.imshow("Canvas", canvas)
    key = cv2.waitKey(1) & 0xFF
    
    # 切換筆的顏色
    if key == ord('b'):
        drawing_color = (0, 0, 0)  # 黑色
    elif key == ord('w'):
        drawing_color = (255, 255, 255)  # 白色
    # 保存圖像
    elif key == ord('s'):
        cv2.imwrite('drawing.png', canvas)
        print("圖片已保存為 drawing.png")
    # 退出
    elif key == 27:  # Esc 鍵
        break

cv2.destroyAllWindows()
