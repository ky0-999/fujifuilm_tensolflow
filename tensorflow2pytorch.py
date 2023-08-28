import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetV2

# 入力データの準備
input_shape = (256, 256, 6)
input_tensor = Input(shape=input_shape)

# EfficientNetV2モデルの構築
base_model = EfficientNetV2.EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=input_tensor)

# 新しい出力層の追加
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output_layer = Dense(1, activation='tanh')(x)  # tanh活性化関数を使用

# 新しいモデルの構築
model = Model(inputs=input_tensor, outputs=output_layer)

# モデルのコンパイル
model.compile(
    loss='mean_squared_error',  # 回帰タスクの損失関数
    optimizer=tf.keras.optimizers.Adam(lr=0.001),  # 最適化アルゴリズムと学習率
    metrics=['mae']  # 平均絶対誤差をモニタリング
)

# モデルのサマリーを表示
model.summary()
