import os,glob,pickle
import tfidf

# 変数初期化
y = []
x = []

# ディレクトリ内のファイル一覧を処理
def read_files(path, label) :
    print("read_files=", path)
    files = glob.glob(path + "/*.txt")
    for f in files :
        # LICENSE.txt以外をファイル追加
        if os.path.basename(f) == 'LICENSE.txt' : continue
        tfidf.add_file(f)
        y.append(label)

# ファイル一覧を読む
read_files('text/kaden-channel',0)
read_files('text/it-life-hack',1)
read_files('text/peachy',2)
read_files('text/topic-news',3)

#TF-IDFベクトルに変換
x = tfidf.calc_files()

# 保存
# パラメータと結果のデータ
pickle.dump([y, x], open('text/genre.pickle', 'wb'))
# 辞書
tfidf.save_dic('text/genre-tfidf.dic')
print('ok')
