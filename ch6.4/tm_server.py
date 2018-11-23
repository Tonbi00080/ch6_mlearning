# サーバー側のプログラム
import json
import flask # webフレームワーク
from flask import request # 値の受け取りのため
import my_text # 文字を４カテゴリに分ける学習プログラム

# ポート番号
TM_PORT_NO = 8085
# HTTPサーバー起動
app = flask.Flask(__name__)
print("http://localhost:" + str(TM_PORT_NO))

# ルートアクセス
@app.route('/', methods = ['GET'])
def index():
    with open("index.html","rb") as f:
        return f.read()

# ./apiアクセス
@app.route('/api', methods = ['GET'])
def api():
    # URLパラメータ取得　request.args.get(キー、デフォルト=なし、タイプ=なし）
    q = request.args.get('q', '')
    if q == '':
        return '{label:"空です", "per":0}'
    print ("q=", q)
    # テキストのジャンル判定
    # check_genre(text) -> ラベル、確率、ラベルのIDを返す
    label, per, no = my_text.check_genre(q)
    # 結果をJSON出力
    return json.dumps({
        "label": label,
        "per": per,
        "genre-no": no
    })
# プログラム名指定で
if __name__ == "__main__":
    #サーバー起動
    app.run(debug = False, port = TM_PORT_NO)
