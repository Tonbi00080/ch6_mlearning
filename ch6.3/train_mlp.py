import pickle
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
import h5py

# 分類するラベルの数
nb_classes = 4

# データベースの読み込み
data = pickle.load(open("text/genre.pickle", "rb"))
y = data[0] # ラベル
x = data[1] # TF-IDF
# ラベルデータをone-hotベクトルに yをnb_classesの数に
y = keras.utils.np_utils.to_categorical(y, nb_classes)
in_size = x[0].shape[0]

# 学習用とテスト用を分割
# 関数も配列を処理するので、分ける形もnumpyの配列に
x_train, x_test, y_train, y_test = train_test_split(
        np.array(x), np.array(y), test_size=0.2)

# MLPモデル構造を定義
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(in_size,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
# 出力
model.add(Dense(nb_classes, activation='softmax'))

# モデルをコンパイル
model.compile(
loss='categorical_crossentropy',
optimizer=RMSprop(),
metrics=['accuracy'])

# 学習を実行
hist = model.fit(x_train, y_train,
batch_size = 128,
epochs = 20,
verbose = 1,
validation_data = (x_test, y_test))

# 評価する
# score[0]：損失、score[1]：正解率
score = model.evaluate(x_test, y_test, verbose=1)
print("正解率=", score[1], 'loss=', score[0])

# 重みデータ保存
model.save_weights('./text/genre-model.hdf5')
open('./text/genre-model.json', 'wt').write(model.to_json())
# 学習経過をグラフに描画
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
