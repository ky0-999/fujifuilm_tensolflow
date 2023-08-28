import cv2
import numpy as np
import random

# 画像Aの読み込み
image_A = cv2.imread('image_A.jpg')

# 背景画像Bの作成
background_size = (256, 256)
background_B = np.zeros((background_size[1], background_size[0], 3), dtype=np.uint8)

# 画像Aを回転する関数
def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

# γ補正を適用する関数
def apply_gamma_correction(image, gamma):
    gamma_corrected = ((image / 255.0) ** gamma) * 255.0
    return np.clip(gamma_corrected, 0, 255).astype(np.uint8)

# 画像Aに緑の線を引く関数
def draw(image):
    center_x = image.shape[1] // 2
    cv2.line(image, (center_x, 0), (center_x, image.shape[0]), (0, 255, 0), 2)

# データ拡張を行う関数
def sample(image):
    rotated_image_A = rotate_image(image, rotation_angle)
    gamma = random.uniform(0.1, 1.0)
    gamma_corrected_image_A = apply_gamma_correction(rotated_image_A, gamma)
    
    x_offset = random.randint(0, background_size[0] - gamma_corrected_image_A.shape[1])
    y_offset = random.randint(0, background_size[1] - gamma_corrected_image_A.shape[0])
    
    direction = random.choice(directions)
    new_x_offset = x_offset + direction[0] * 50
    new_y_offset = y_offset + direction[1] * 50
    
    background_B[new_y_offset:new_y_offset+gamma_corrected_image_A.shape[0], new_x_offset:new_x_offset+gamma_corrected_image_A.shape[1]] = gamma_corrected_image_A
    
    draw(background_B)  # 画像Aに線を引く

# 画像Aの中心から8方向に平行移動するための方向リスト
directions = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

# 10枚の合成画像を生成
num_images = 10
rotation_angle = 45  # 回転角度（例として45度とします）

for i in range(num_images):
    sample(image_A)

# 合成された画像の表示
cv2.imshow('Composite Images', background_B)
cv2.waitKey(0)
cv2.destroyAllWindows()
