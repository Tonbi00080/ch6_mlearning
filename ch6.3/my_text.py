import pickle, tfidf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import model_from_json


# テキスト設定
text1 = """新しい冷蔵庫と炊飯器が発売されました。新生活を始めたいあなたに
おすすめです。セールは１２月までです。"""

text2 = """システムエンジニアの派遣には法律の問題が大きく絡みます。プロジェ
クトを立ち上げる時は注意が必要です。"""

text3 = """トランプ大統領がホワイトハウスの記者を出入り禁止にしましたが、
発言回数の制限等を設け、解除されました。"""

#TF-IDFの辞書を読み込み
tfidf.load_dic("text/genre-tfidf.dic")

#train_mlpで作成したもの
# KerasのモデルをJSONで読み込む
model = model_from_json(open('./text/genre-model.json').read())
# Kerasのモデルを定義して、重みデータを読み込む
model.load_weights('./text/genre-model.hdf5')

# テキスト判定
def check_genre(text):
    # ラベルの定義
    LABELS = ["家電","IT","恋愛","ニュース"]
    # TF-IDFのベクトルに変換
    data = tfidf.calc_text(text)
    # MLPで予測
    pre = model.predict(np.array([data]))[0]
    n = pre.argmax()
    print(LABELS[n], "(", pre[n], ")")
    return LABELS[n], float(pre[n]), int(n)


if __name__ == '__main__' :
    check_genre(text1)
    check_genre(text2)
    check_genre(text3)
