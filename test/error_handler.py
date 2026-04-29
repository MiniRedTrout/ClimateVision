from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def health():
    return "OK"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # КЛЮЧЕВОЙ МОМЕНТ: хост должен быть "0.0.0.0"
    app.run(host="0.0.0.0", port=port)