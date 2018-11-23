import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import numpy as np

# TF-IDFのDB読み込む
data = pickle.load(open("text/genre.pickle", "rb"))
y = data[0] # label
x = data[1] # TF-IDF

# 学習用とテスト用に分ける
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =  0.2)

# ナイーブベイズで学習
model = GaussianNB()
model.fit(x_train, y_train)

# 評価して結果出力
y_pred = model.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
rep = metrics.classification_report(y_test, y_pred)

print("正解率：",acc)
print(rep)
