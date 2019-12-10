import pickle
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
        audio_array = pickle.loads(data)
        if isinstance(audio_array, (list, tuple)):
            audio_array, granularity = audio_array
        else:
            audio_array, granularity = audio_array, None
        buffer_read_time = time.time() - now
        now = time.time()
        result = predictor.run(audio_array, granularity=granularity)
        prediction_time = time.time() - now
        return json.dumps({
            "result":[i.tolist() for i in result],
            "prediction_time": prediction_time,
            "buffer_read_time": buffer_read_time,
            "request_read_time": request_read_time,
            "granularity": 1 if granularity is None else granularity
        })
    else:
        return "415 Unsupported Media Type"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
