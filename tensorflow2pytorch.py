import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

input_shape = (256, 256, 6)  # 入力画像サイズとチャンネル数
output_units = 1  # 出力層のユニット数

# 入力層を定義
input_layer = Input(shape=input_shape)

# EfficientNetV2モデルを読み込む
effnet_model = EfficientNetV2B0(input_tensor=input_layer, include_top=False, weights='imagenet')

# プーリング層などを含むEfficientNetV2の出力をフラットにする
flatten_layer = tf.keras.layers.Flatten()(effnet_model.output)

# 回帰のための出力層を定義
output_layer = Dense(output_units, activation='linear')(flatten_layer)

# モデルを結合
model = Model(inputs=input_layer, outputs=output_layer)

# モデルをコンパイル
model.compile(optimizer='adam', loss='mean_squared_error')

# モデルのサマリを表示
model.summary()

# モデルをトレーニング
# model.fit(...)
