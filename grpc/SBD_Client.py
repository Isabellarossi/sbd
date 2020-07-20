#============================================================
# import packages
#============================================================
from concurrent import futures
import grpc
import cv2
import BagAnalysis_pb2
import BagAnalysis_pb2_grpc

import base64
import sys
import pyrealsense2 as rs
import numpy as np


#============================================================
# property
#============================================================
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# pipeline.start(config)


# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#============================================================
# functions
#============================================================
def Request(frame):

	#print("origin size : ", sys.getsizeof(gray))
	ret, buf = cv2.imencode('.jpg', frame)

	if ret != 1:
			return

	# encode to base64
	b64e = base64.b64encode(buf)
	#print("base64 encode size : ", sys.getsizeof(b64e))

	yield BagAnalysis_pb2.AnalysisRequest(image = b64e)


#====================
def run():

	channel = grpc.insecure_channel('localhost:50051')
	stub = BagAnalysis_pb2_grpc.BagAnalysisStub(channel)
	
	# while True:

	try:
	
		# ret, frame = cap.read()
		# if ret != 1:
		# 	continue
		# frames = pipeline.wait_for_frames()
		# color_frame = frames.get_color_frame()
		# if not color_frame:
		# 	continue
		# frame = np.asanyarray(color_frame.get_data())
		# print("start.....")
		frame = cv2.imread("/home/don/Pictures/suitcase.jpg")


		# cv2.imshow('Capture Image', frame)
		# k = cv2.waitKey(1)
		# if k == 27:
		# 	break

		responses = stub.Analyse( Request(frame) )
		# print(responses)
		for res in responses:
			# print("hello")
			print(res)
	
	except grpc.RpcError as e:
		print(e.details())
			#break



#============================================================
# Awake
#============================================================



#============================================================
# main
#============================================================
if __name__ == '__main__':
	run()



#============================================================
# after the App exit
#============================================================
# cap.release()
# pipeline.stop()

cv2.destroyAllWindows()