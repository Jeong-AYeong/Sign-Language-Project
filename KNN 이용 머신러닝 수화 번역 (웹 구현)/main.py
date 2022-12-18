# flask web
import time
import numpy as np
from image_process import gen_frames
from flask import Flask, render_template, Response

# flask web
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# run
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8000)
