import os
import sys
import numpy as np
import predict_audio
import pyaudio

RATE = 48000
CHUNK = int(RATE / 5)


def floating_to_int(number_string):
    number_string = str(number_string)
    if number_string[-4] != "e":
        result = float(number_string)
        result *= 100
        result = round(result, 3);
        return result
    else:
        power = int(float(number_string[-2:]))
        n = float(number_string[:5])
        temp = 1 / (10 ** power)
        result = round(n * temp, 5)
        result = result * 100
        result = round(result, 3)
        return result

if __name__ == "__main__":
    predictor = predict_audio.Predictor(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models/vad/model.ckpt-200106"
        ))
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    arr = bytearray()
    for i in range(sys.maxsize ** 10):
        arr += stream.read(CHUNK)
        if len(arr) >= 192000:
            #print(predictor.run_from_raw(arr))
            result = predictor.run_from_raw(arr)
            #print(result)
            print({"noise": floating_to_int(result[0]), "voice": floating_to_int((result[1]))})

            arr = bytearray()
        # print(np.frombuffer(stream.read(CHUNK), dtype=np.float32))
    stream.stop_stream()
    stream.close()
    p.terminate()
