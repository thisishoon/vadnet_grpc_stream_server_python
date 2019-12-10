import argparse
import sys, os, csv, glob, random, threading, time, enum, json
from typing import Optional, List
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import librosa as lr
import vadnet.utils as utils


DEFAULT_CKPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models/vad/model.ckpt-200106"
)


class Predictor:
    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CKPT_PATH,
        additional_layer_names=None,
    ):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = os.path.split(checkpoint_path)[0]
        self.graph = tf.get_default_graph()
        self.additional_layer_names = additional_layer_names
        self.saver = tf.train.import_meta_graph(self.checkpoint_path + ".meta")
        with open(os.path.join(self.checkpoint_dir, "vocab.json"), "r") as fp:
            self.vocab = json.load(fp)

        self.x = self.graph.get_tensor_by_name(self.vocab["x"])
        self.sr = int(self.x.shape[1])  # 48000
        self.y = self.graph.get_tensor_by_name(self.vocab["y"])
        self.init = self.graph.get_operation_by_name(self.vocab["init"])
        self.logits = self.graph.get_tensor_by_name(self.vocab["logits"])
        self.ph_n_shuffle = self.graph.get_tensor_by_name(
            self.vocab["n_shuffle"]
        )
        self.ph_n_repeat = self.graph.get_tensor_by_name(
            self.vocab["n_repeat"]
        )
        self.ph_n_batch = self.graph.get_tensor_by_name(self.vocab["n_batch"])
        self.layers = [self.logits]
        if self.additional_layer_names:
            for layer_name in self.additional_layer_names:
                self.layers.append(self.graph.get_tensor_by_name(layer_name))
        self.sess = tf.Session()
        self.saver.restore(self.sess, checkpoint_path)

    def run(self, audio_array, n_batch=1, granularity=None):
        result = [
            np.empty([0] + x.shape[1:].as_list(), dtype=np.float32)
            for x in self.layers
        ]
        frames = utils.audio_to_frames(audio_array, self.sr, None)
        if granularity is not None:
            frames = [frames]
            for gran in range(1, granularity):
                audarr = audio_array[gran * int(round(self.sr / granularity)):]
                n_add = self.sr * len(frames[0]) - len(audarr)
                if n_add > 0:
                    audarr = np.concatenate([audarr, np.zeros_like(audarr[:n_add])])
                add = utils.audio_to_frames(audarr, self.x.shape[1], None)
                frames += [add]
            print([i.shape for i in frames])
            frames = np.stack(frames, 1).reshape(-1, 48000)
        labels = np.zeros((frames.shape[0],), dtype=np.int32)
        self.sess.run(
            self.init,
            feed_dict={
                self.x: frames,
                self.y: labels,
                self.ph_n_shuffle: 1,
                self.ph_n_repeat: 1,
                self.ph_n_batch: n_batch if n_batch > 0 else frames.shape[0],
            },
        )

        while True:
            try:
                outputs = self.sess.run(self.layers)
                for i, output in enumerate(outputs):
                    result[i] = np.concatenate([result[i], output])
            except tf.errors.OutOfRangeError:
                break
        return result

    def run_from_file(self, audio_path, n_batch=1, granularity=None):
        audio = utils.audio_from_file(audio_path, 48000)
        return self.run(audio, n_batch, granularity)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Use NetVad to process a .wav file."
    )
    parser.add_argument(
        "audio_filepath",
        metavar="A",
        type=str,
        help="path to the .wav file to process",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=DEFAULT_CKPT_PATH,
        help=(
            "path to the checkpoint file (excluding extension,"
            " e.g. 'models/vad/model.ckpt-200106')"
        ),
    )
    args = parser.parse_args()

    predictor = Predictor(args.ckpt)

    result = predictor.run_from_file(args.audio_filepath)
    print(result, flush=False)
