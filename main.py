import os
from flask import Flask, request, render_template, jsonify
import whisper
import tempfile

app = Flask(__name__, template_folder='.')

# Whisperモデルを最初に一度だけロードする
# アプリケーションの起動は少し遅くなるが、リクエストごとの処理は速くなる
try:
    model = whisper.load_model("base")
except Exception as e:
    print(f"Whisperモデルのロード中にエラーが発生しました: {e}")
    model = None

@app.route('/')
def index():
    """HTMLページを表示する"""
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """音声ファイルを文字起こしする"""
    if model is None:
        return jsonify({"error": "Whisperモデルがロードされていません。"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "ファイルがありません"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "ファイルが選択されていません"}), 400

    # 一時ファイルに音声を保存して処理する
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            file.save(tmp.name)
            result = model.transcribe(tmp.name)
            transcribed_text = result["text"]
        
        os.remove(tmp.name) # 一時ファイルを削除
        
        return jsonify({"text": transcribed_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # `ffmpeg` が見つからないエラー対策として、ffmpeg.exeがあるパスを一時的に環境変数に追加
    # このスクリプトと同じディレクトリにffmpeg.exeを置くことを想定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in os.environ['PATH']:
        os.environ['PATH'] = os.path.join(script_dir, '') + os.pathsep + os.environ['PATH']

    # アプリケーションの起動
    # host='0.0.0.0' とすることで、同じネットワーク内の他のPCからもアクセス可能になる
    app.run(host='0.0.0.0', port=5001, debug=True)