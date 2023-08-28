import cv2
import numpy as np

# 配列Aに3枚の画像を代入
array_A = [cv2.imread('image1.jpg'), cv2.imread('image2.jpg'), cv2.imread('image3.jpg')]

# リサイズ後の画像を格納するための空の配列Bを用意
array_B = []

# リサイズの幅と高さを指定
new_width = 300
new_height = 200

# 配列Aの画像をリサイズして配列Bに追加
for img in array_A:
    resized_img = cv2.resize(img, (new_width, new_height))
    array_B.append(resized_img)

# 配列Bの画像を表示
for resized_img in array_B:
    cv2.imshow('Resized Image', resized_img)
    cv2.waitKey(0)

# ウィンドウをすべて閉じる
cv2.destroyAllWindows()
