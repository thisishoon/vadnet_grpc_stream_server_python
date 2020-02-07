# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc
import media_pb2 as media__pb2
import predict_audio
import concurrent.futures as futures
import numpy as np
import grpc
import media_pb2_grpc
import time, os


class MediaServiceServicer(media_pb2_grpc.MediaServiceServicer):

    def __init__(self, predictor):
        self.predictor = predictor

    def filter(self, request_iterator, context):
        for request in request_iterator:
            data = self.predictor.run_from_raw(request.raw)
            data = data[0].T
            # print(data)
            yield media__pb2.inferredResult(noise=data[0], voice=data[1])



def serve(predictor):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    media_pb2_grpc.add_MediaServiceServicer_to_server(
        MediaServiceServicer(predictor), server)
    server.add_insecure_port('[::]:9999')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    predictor = predict_audio.Predictor(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models/vad/model.ckpt-200106"
        ))
    serve(predictor)
