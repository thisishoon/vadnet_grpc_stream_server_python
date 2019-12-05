import argparse
import sys, os, csv, glob, random, threading, time, enum, json
from typing import Optional, List
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import librosa as lr
import youtube_dl


def audio_from_file(path, sr, ext="", root="", offset=0.0, duration=None):
    path = os.path.join(root, "{}{}".format(path, ext))
    try:
        audio, _ = lr.load(
            path,
            sr=sr,
            mono=True,
            offset=offset,
            duration=duration,
            dtype=np.float32,
            res_type="kaiser_fast",
        )
        audio.shape = (-1, 1)
        return audio
    except ValueError as ex:
        print("value error {}\n{}".format(path, ex))
        return []
    except Exception as ex:
        print("could not read {}\n{}".format(path, ex))
        return None


def audio_to_frames(x, n_frame, n_step=None):
    if n_step is None:
        n_step = n_frame
    if len(x.shape) == 1:
        x.shape = (-1, 1)
    n_overlap = n_frame - n_step
    n_frames = (x.shape[0] - n_overlap) // n_step
    n_keep = n_frames * n_step + n_overlap
    strides = list(x.strides)
    strides[0] = strides[1] * n_step
    return np.lib.stride_tricks.as_strided(
        x[0:n_keep, :], (n_frames, n_frame), strides
    )


def get_model_name(path: str):
    if os.path.isdir(path):
        path = tf.train.get_checkpoint_state(path).model_checkpoint_path
    return path


def print_checkpoint(path: str, name: str):
    path = get_model_name(path)
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    if not name:
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            tensor = reader.get_tensor(key)
            print("{}:{}".format(key, tensor.shape))
            print(tensor)
    else:
        print(name)
        print(reader.get_tensor(name))


def print_graph():

    vars = tf.global_variables()
    for var in vars:
        print("{}:{}".format(var.name, var.eval().shape))
        print(var.eval())


def get_var_from_checkpoint(path: str, tensor_name: str):
    path = get_model_name(path)
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    return reader.get_tensor(tensor_name)


def get_var_from_graph(name: str):
    vars = tf.global_variables()
    for var in vars:
        if var.name == name:
            return var
    return None


def update_var_in_graph(sess: tf.Session, name: str, value: np.ndarray):
    var = get_var_from_graph(name)
    sess.run(var.assign(value))


def update_var_from_checkpoint(
    sess: tf.Session, name_to: str, name_from: str, path: str
):
    path = get_model_name(path)
    var_from = get_var_from_checkpoint(path, name_from)
    update_var_in_graph(sess, name_to, var_from)


def predict_from_checkpoint(
    audio: np.ndarray,
    checkpoint_path: str,
    additional_layer_names=None,
    n_batch=1,
) -> List:

    result = None
    checkpoint_dir = os.path.split(checkpoint_path)[0]

    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(checkpoint_path + ".meta")
        with open(os.path.join(checkpoint_dir, "vocab.json"), "r") as fp:
            vocab = json.load(fp)

        x = graph.get_tensor_by_name(vocab["x"])
        y = graph.get_tensor_by_name(vocab["y"])
        init = graph.get_operation_by_name(vocab["init"])
        logits = graph.get_tensor_by_name(vocab["logits"])
        ph_n_shuffle = graph.get_tensor_by_name(vocab["n_shuffle"])
        ph_n_repeat = graph.get_tensor_by_name(vocab["n_repeat"])
        ph_n_batch = graph.get_tensor_by_name(vocab["n_batch"])
        layers = [logits]
        if additional_layer_names:
            for layer_name in additional_layer_names:
                layers.append(graph.get_tensor_by_name(layer_name))
        result = [
            np.empty([0] + x.shape[1:].as_list(), dtype=np.float32)
            for x in layers
        ]
        frames = audio_to_frames(audio, x.shape[1], None)
        labels = np.zeros((frames.shape[0],), dtype=np.int32)

        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            sess.run(
                init,
                feed_dict={
                    x: frames,
                    y: labels,
                    ph_n_shuffle: 1,
                    ph_n_repeat: 1,
                    ph_n_batch: n_batch if n_batch > 0 else frames.shape[0],
                },
            )

            count = 0
            while True:
                try:
                    outputs = sess.run(layers)
                    for i, output in enumerate(outputs):
                        result[i] = np.concatenate([result[i], output])
                    # labels[count:count+output.shape[0]] = np.argmax(output, axis=1)
                    # count += output.shape[0]
                except tf.errors.OutOfRangeError:
                    break
    return result


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
        default="models/vad/model.ckpt-200106",
        help=(
            "path to the checkpoint file (excluding extension,"
            " e.g. 'models/vad/model.ckpt-200106')"
        ),
    )
    args = parser.parse_args()

    audio = audio_from_file(args.audio_filepath, 48000)
    result = predict_from_checkpoint(
        audio, args.ckpt, additional_layer_names=None, n_batch=1
    )
    print(result)
