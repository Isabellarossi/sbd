
from threading import Thread
from concurrent import futures
import sys, os, time
# print(sys.path)

import grpc
import BagAnalyserService_pb2_grpc
import BagAnalyserService_pb2

# from demo.settings import *
from demo.settings import *
import demo.settings as settings
from demo.demo_w_gRPC import AI_callback, AnalyseResponse_data, efficientDet_pred, sbd_pred

# import google.protobuf.timestamp_pb2


# SERVER_ADDRESS = 'localhost:23333'
SERVER_ADDRESS = '172.29.25.99:23333'

SERVER_ID = 1


class BagAnalysisService(BagAnalyserService_pb2_grpc.BagAnalysisServiceServicer):

        # Starts a new analysis session between client and server.
        def BeginSession(self, BeginSessionRequest, context):
            print("BeginSession")
            Response_data = 1
            response = BagAnalyserService_pb2.BeginSessionResponse(
                status=Response_data)
            return response
            # return (BeginSessionResponse) 

        #  Ends current session
        def EndSession(self, ClientIdentifier, context):
            print("EndSession")
            Response_data = 1
            response = BagAnalyserService_pb2.EndSessionResponse(
                status=Response_data)
            
            return response
            # return (EndSessionResponse)

        #  Starts 
        def StartAnalyse(self, AnalyseRequest, context):
            # google.protobuf.Timestamp 
            # AI_callback()
            settings.startAnalyse = True
            settings.cameraId = 1
            res = AnalyseResponse_data()
            def response_messages():
                
                while(settings.startAnalyse):
                    
                    res = sbd_pred() # AI_callback()
                    print('Start Analysis res.....', res.cameraId, res.status,res.rejectReasons, res.classifications)
                    from google.protobuf.timestamp_pb2 import Timestamp
                    timestamp = Timestamp()
                    timestamp.GetCurrentTime()
                    # print(timestamp)
                    response = BagAnalyserService_pb2.AnalyseResponse(

                        status = res.status,
                        rejectReasons = res.rejectReasons, #["multi bag", "extended_handle"]
                        # cameraId = res.cameraId,           # int32 cameraId = 4;
                        cameraId = settings.cameraId,
                        timestamp = timestamp,  # google.protobuf.Timestamp timestamp = 5;
                        processingTime = res.processingTime,     # int32 processingTime = 6;
                        length = res.length,             # int32 length = 7;
                        width = res.width,              # int32 width = 8;
                        height = res.height,             # int32 height = 9;
                        placementStatus =  False, # bool placementStatus = 10;
                        placementDetails = 0,   # int32 placementDetails = 11;
                        orientationStatus = True, # bool orientationStatus = 12;
                        orientaionDetails = 0,  # int32 orientaionDetails =13;
                        classifications = res.classifications, # ["suitcase", "soft_bag"], # string classification=14;
                        components = res.components, #[""],        # repeated string components = 15;
                        boundingBoxX1 = res.boundingBoxX1,      # int32 15
                        boundingBoxY1 = res.boundingBoxY1,
                        boundingBoxX2 = res.boundingBoxX2,
                        boundingBoxY2 = res.boundingBoxY2,
                        objectId  = res.objectId,
                        
                        )
                    if res.status == 3: # rejected
                        # settings.startAnalyse = False
                        print("Rejected case.")

                    yield response

                return response
            return response_messages()


        def StopAnalyse(self, ClientIdentifier, context):
            def response_messages():
                Response_data = 1
                response = BagAnalyserService_pb2.StopAnalyseResponse(
                    status=Response_data)
                    
                settings.startAnalyse = False
                print("Stopped Analysis")
                return response
            # return (StopAnalyseResponse) 
            return response_messages()

        # Performs a single analysis
        def Analyse(self, AnalyseRequest, context):
            
            def response_messages():
                settings.cameraId = 2
                res = sbd_pred() #AI_callback()

                print('Analysis res.....', res.cameraId, res.status,res.rejectReasons, res.classifications)
                from google.protobuf.timestamp_pb2 import Timestamp
                timestamp = Timestamp()
                timestamp.GetCurrentTime()
                print("res.rejectReasons,res.classification, res.components...:", res.rejectReasons,res.classifications, res.components)
                response = BagAnalyserService_pb2.AnalyseResponse(

                    status = res.status,            # 1
                    rejectReasons = res.rejectReasons,# 2 eg. ["multi bag", "extended_handle"]
                    cameraId = settings.cameraId, # 3
                    timestamp = timestamp,  # google.protobuf.Timestamp timestamp = 4;
                    processingTime = res.processingTime ,     # int32 processingTime = 5;
                    length = res.length,             # int32 length = 6;
                    width = res.width,              # int32 width = 7;
                    height = res.height,             # int32 height = 8;
                    placementStatus =  False, # bool placementStatus = 9;
                    placementDetails = 0,   # int32 placementDetails = 10;
                    orientationStatus = True, # bool orientationStatus = 11;
                    orientaionDetails = 0,  # int32 orientaionDetails =12;
                    classifications = res.classifications, #res.classification, # string classification=13;
                    components = res.components,# repeated string components = 14;
                    boundingBoxX1 = res.boundingBoxX1,      # int32 15
                    boundingBoxY1 = res.boundingBoxY1,
                    boundingBoxX2 = res.boundingBoxX2,
                    boundingBoxY2 = res.boundingBoxY2,
                    objectId  = res.objectId,
                    
                    )
                return response
            return response_messages()

        # Ping function
        def Ping(self, ClientIdentifier, context):
            print("Ping....ClientIdentifier: ", ClientIdentifier)
            PingResponse = 1
            response = BagAnalyserService_pb2.StopAnalyseResponse(
                status=PingResponse)
            return response


def start_grpc_server():
    server = grpc.server(futures.ThreadPoolExecutor())

    BagAnalyserService_pb2_grpc.add_BagAnalysisServiceServicer_to_server(BagAnalysisService(), server)

    server.add_insecure_port(SERVER_ADDRESS)
    print("------------------start Python GRPC server")
    server.start()
    import socket
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    print("Your Computer Name is:" + hostname)
    print("Your Computer IP Address is:" + IPAddr)

    server.wait_for_termination()


    # If raise Error:
    #   AttributeError: '_Server' object has no attribute 'wait_for_termination'
    # You can use the following code instead:
    # import time
    # while 1:
    #     time.sleep(10) 
    


if __name__ == '__main__':

    # start_time = time.time()
    # message_txt = AI_callback()
    # message_txt = sbd_pred()
    # stop_time = time.time()
    # print("AI_callback time:", stop_time - start_time)
    # test_efficientDet()


    start_grpc_server()






