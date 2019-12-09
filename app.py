import numpy as np
from flask import Flask, request
from vadnet.predict_audio import Predictor
import json


predictor = Predictor()
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def api_message():

    if request.headers["Content-Type"] == "application/octet-stream":
        audio_array = np.frombuffer(request.data, dtype="float32").reshape(
            -1, 1
        )
        print(audio_array.shape)
        result = predictor.run(audio_array)
        return json.dumps([i.tolist() for i in result])

    else:
        return "415 Unsupported Media Type"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
