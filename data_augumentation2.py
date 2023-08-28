import cv2
import numpy as np
import random

# 画像Aの読み込み
image_A = cv2.imread('image_A.jpg')

# 背景画像Bの作成
background_size = (256, 256)
background_B = np.zeros((background_size[1], background_size[0], 3), dtype=np.uint8)

# 背景画像Cの生成
background_C = np.random.randint(0, 256, background_B.shape, dtype=np.uint8)

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

# 画像Aを回転して背景Cに張り付ける
rotation_angle = 45  # 回転角度（例として45度とします）
rotated_image_A = rotate_image(image_A, rotation_angle)

# 10枚の合成画像を生成
num_images = 10
for i in range(num_images):
    gamma = random.uniform(0.1, 1.0)  # ランダムなγ値を選択
    gamma_corrected_image_A = apply_gamma_correction(rotated_image_A, gamma)
    
    x_offset = random.randint(0, background_size[0] - gamma_corrected_image_A.shape[1])
    y_offset = random.randint(0, background_size[1] - gamma_corrected_image_A.shape[0])
    
    # 中心から8方向にランダムに平行移動する
    directions = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    direction = random.choice(directions)
    new_x_offset = x_offset + direction[0] * 50
    new_y_offset = y_offset + direction[1] * 50
    
    # 画像Aと背景Cをアルファブレンドで合成
    alpha = 0.5  # 透明度の割合（例として0.5とします）
    blended_image = cv2.addWeighted(gamma_corrected_image_A, alpha, background_C[new_y_offset:new_y_offset+gamma_corrected_image_A.shape[0], new_x_offset:new_x_offset+gamma_corrected_image_A.shape[1]], 1 - alpha, 0)
    
    background_B[new_y_offset:new_y_offset+gamma_corrected_image_A.shape[0], new_x_offset:new_x_offset+gamma_corrected_image_A.shape[1]] = blended_image

# 合成された画像の表示
cv2.imshow('Composite Images', background_B)
cv2.waitKey(0)
cv2.destroyAllWindows()
