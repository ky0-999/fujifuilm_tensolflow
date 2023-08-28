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



import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import EfficientNetV2


# 入力データの準備
input_shape = (256, 256, 6)
input_tensor = Input(shape=input_shape)

# EfficientNetV2モデルの構築
model = tf.keras.Sequential()
model.add(input_tensor)  # 入力層の追加

# EfficientNetV2 v0モデルの追加
model.add(EfficientNetV2.EfficientNetV2B0(include_top=True, weights=None, input_tensor=input_tensor, classes=1))

# モデルのコンパイル
model.compile(
    loss='mean_squared_error',  # 回帰タスクの損失関数
    optimizer=tf.keras.optimizers.Adam(lr=0.001),  # 最適化アルゴリズムと学習率
    metrics=['mae']  # 平均絶対誤差をモニタリング
)

# モデルのサマリーを表示
model.summary()

