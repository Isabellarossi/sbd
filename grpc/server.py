
from threading import Thread
from concurrent import futures
import sys, os
# print(sys.path)

import grpc
import demo_pb2_grpc
import demo_pb2

# from demo.settings import *
from demo.settings import *
import demo.settings as settings
from demo.demo_w_gRPC import * #AI_callback



SERVER_ADDRESS = 'localhost:23333'
# SERVER_ADDRESS = '172.29.25.56:50052'
SERVER_ID = 1



class DemoServer(demo_pb2_grpc.GRPCDemoServicer):


    # unary-unary(In a single call, the client can only send request once, and the server can
    # only respond once.)
    def SimpleMethod(self, request, context):
        print("SimpleMethod called by client(%d) the message: %s" %
              (request.client_id, request.request_data))

        response = demo_pb2.Response(
            server_id=SERVER_ID,
            response_data="Python server SimpleMethod Ok!!!!")
        return response

    # stream-unary (In a single call, the client can transfer data to the server several times,
    # but the server can only return a response once.)
    def ClientStreamingMethod(self, request_iterator, context):
        print("ClientStreamingMethod called by client...")
        for request in request_iterator:
            print("recv from client(%d), message= %s" %
                  (request.client_id, request.request_data))
        response = demo_pb2.Response(
            server_id=SERVER_ID,
            response_data="Python server ClientStreamingMethod ok")
        return response

    # unary-stream (In a single call, the client can only transmit data to the server at one time,
    # but the server can return the response many times.)
    def ServerStreamingMethod(self, request, context):
        print("ServerStreamingMethod called by client(%d), message= %s" %
              (request.client_id, request.request_data))

        if request.request_data == "start":
            settings.force_in_button = False
            message_txt = "auto mode: start"
        elif request.request_data == "check" :
            settings.check_button = True
            message_txt = "manual mode: check"
        else:
            print("error: unkown command.")


        # create a generator
        def response_messages():
            # for i in range(5):
            #     response = demo_pb2.Response(
            #         server_id=SERVER_ID,
            #         response_data=("send by Python server, message=%d" % i))
            #     yield response
            automode = True
            # while(automode == True):
            for i in range(10):
                time.sleep(1)
                # AI_callback()
                message_txt = settings.event_list
                if message_txt == None:
                    message_txt.append("detect result is ok")
                else:
                    print("messages:", message_txt)
                # message_txt = "hello"
                response = demo_pb2.Response(
                    server_id=SERVER_ID,
                    response_data=("send by Python server, message=%s" % str(message_txt)))
                yield response

        return response_messages()


    # stream-stream (In a single call, both client and server can send and receive data
    # to each other multiple times.)
    def BidirectionalStreamingMethod(self, request_iterator, context):
        print("BidirectionalStreamingMethod called by client...")
        stop_command = False
        
 
        # Open a sub thread to receive data
        def parse_request():
            for request in request_iterator:
                print("recv from client(%d), message= %s" %
                      (request.client_id, request.request_data))
                if request.request_data == "start":
                    print("start analysis..")
                    settings.force_in_button = False
                    message_txt = "auto mode: start"
                    # while(1):
                    #     event = AI_callback()
                    #     # settings.event_list.append("1")
                    #     if settings.event_list >0:
                    #         print("event!!!!!!!!!!!")
                    #         break
                    #     yield demo_pb2.Response(
                    #         server_id=SERVER_ID,
                    #         response_data=("send by Python server, message= %s" % str(time.time())))

                elif request.request_data == "check":
                    print("checking....")
                    settings.check_button = True
                    message_txt = "manual mode: check"
                    # AI_callback()
                    sbd_pred()
                
                elif request.request_data == "stop":
                    print("checking....")
                    stop_command = True
                    message_txt = "manual mode: check"
                    # AI_callback()
                    sbd_pred()
                else:
                    print("unkonw command:", request.request_data )


        t = Thread(target=parse_request)
        t.start()

        while(1):

            yield demo_pb2.Response(
                server_id=SERVER_ID,
                response_data=("send by Python server, message= %s" % str(time.time())))
            time.sleep(2)

        t.join()


def start_grpc_server():
    server = grpc.server(futures.ThreadPoolExecutor())

    demo_pb2_grpc.add_GRPCDemoServicer_to_server(DemoServer(), server)

    server.add_insecure_port(SERVER_ADDRESS)
    print("------------------start Python GRPC server")
    server.start()
    server.wait_for_termination()


    # If raise Error:
    #   AttributeError: '_Server' object has no attribute 'wait_for_termination'
    # You can use the following code instead:
    # import time
    # while 1:
    #     time.sleep(10) 
    


if __name__ == '__main__':

    # message_txt = AI_callback()
    # print("message:",message_txt)
    start_grpc_server()
