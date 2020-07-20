
from threading import Thread
from concurrent import futures
import sys, os, time
import cv2, base64
import numpy as np
# print(sys.path)
sys.path.append("..")
# import Image, io
import grpc
import BagAnalysis_pb2_grpc
import BagAnalysis_pb2
# from BagAnalysis_pb2_grpc import BagAnalysisServicer

# # from demo.settings import *
# from demo.settings import *
# import demo.settings as settings
# from demo.demo_w_gRPC import AI_callback, AnalyseResponse_data, efficientDet_pred, sbd_pred

# import google.protobuf.timestamp_pb2


SERVER_ADDRESS = '[::]:50051'
# SERVER_ADDRESS = '172.29.25.56:23333'
# SERVER_ADDRESS = '172.29.25.56:50051'

SERVER_ID = 1


class Analyse(BagAnalysis_pb2_grpc.BagAnalysisServicer):
        # streaming image
        def Analyse(self, request_iterator, context):

            for req in request_iterator:

                print(time.time())
                print("req", req.img)
                # image = Image.open(io.BytesIO(req.img))
                # image.save(image.jpg)

                print("done")
                with open("image.jpg", "wb") as imageA:
                    imageA.write(req.img)

                bb = np.array(req.img)

                frame = np.array(list(req.img))
                print(frame.shape)
                frame = frame.reshape( (576,704) )
                frame = np.array(frame, dtype = np.uint8 )
        
                processed = frame
                #display processed video
                cv2.imshow('Processed Image', processed)
                cv2.waitKey(1)
            
                #This line will send a value back to the client for each frame.
                yield BagAnalysis_pb2.AnalysisResponse(result = 1,  flags = [1],)
                # yield imageTest_pb2.MsgReply(reply = ppl_counter.people_count1 )


def start_grpc_server():
    server = grpc.server(futures.ThreadPoolExecutor())

    BagAnalysis_pb2_grpc.add_BagAnalysisServicer_to_server(Analyse(), server)

    server.add_insecure_port(SERVER_ADDRESS)
    print("------------------start Python GRPC server")
    server.start()
    import socket
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    print("Your Computer Name is:" + hostname)
    print("Your Computer IP Address is:" + IPAddr)
    server.wait_for_termination()



if __name__ == '__main__':

    # start_time = time.time()

    # message_txt = sbd_pred()
    # stop_time = time.time()
    # print("AI_callback time:", stop_time - start_time)

    start_grpc_server()






