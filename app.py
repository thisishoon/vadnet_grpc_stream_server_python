import numpy as np
from flask import Flask, request
from vadnet.predict_audio import Predictor
import json
import time


predictor = Predictor()
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def api_message():

    if request.headers["Content-Type"] == "application/octet-stream":
        now = time.time()
        data = request.data
        request_read_time = time.time() - now
        now = time.time()
        audio_array = np.frombuffer(data, dtype="float32")
        buffer_read_time = time.time() - now
        now = time.time()
        result = predictor.run(audio_array.reshape(-1, 1))
        prediction_time = time.time() - now
        return json.dumps({
            "result":[i.tolist() for i in result],
            "prediction_time": prediction_time,
            "buffer_read_time": buffer_read_time,
            "request_read_time": request_read_time,
        })

    else:
        return "415 Unsupported Media Type"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
