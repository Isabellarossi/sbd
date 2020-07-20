#============================================================
# import packages
#============================================================
from concurrent import futures
import time
import cv2
import grpc
import base64
import numpy as np
import BagAnalysis_pb2
import BagAnalysis_pb2_grpc
import sys

sys.path.append("..")
# from demo.settings import *
from prediction.settings import *
import prediction.settings as settings
from prediction.demo_w_gRPC import AI_callback, AnalyseResponse_data, efficientDet_pred, sbd_pred

#============================================================
# classes
#============================================================
class ShowVideoStream:
	
	img = None
	thread = futures.ThreadPoolExecutor(max_workers=1)

	#==========
	def start(self):
		self.thread.submit(self.ShowWindow)

	#==========
	def set(self, img):
		self.img = img

	#==========
	def ShowWindow(self):
		while True:
			if self.img is not None:
				cv2.imshow('dst Image', self.img)
				k = cv2.waitKey(1)
				if k == 27:
					break




#====================
class Greeter(BagAnalysis_pb2_grpc.BagAnalysisServicer):

	#==========
	def __init__(self):
		pass

	#==========
	def Analyse(self, request_iterator, context):

		timer = 0
		print("start..")
		for req in request_iterator:
		
			print('process time = ' + str(time.clock() - timer))
			timer = time.clock()
			
			# decode from base64
			b64d = base64.b64decode(req.image)
			#print("base64 decode size : ", sys.getsizeof(b64d))
			
			# base64 buffer to uint8
			dBuf = np.frombuffer(b64d, dtype = np.uint8)
			#print("buffer size : ", sys.getsizeof(dBuf))
			
			# decode to cv2
			dst = cv2.imdecode(dBuf, cv2.IMREAD_COLOR)
			#print("dst size : ", sys.getsizeof(dst))
			
			# set pixels
			# show.set(dst)
			response = sbd_pred(dst)

			# success
			yield BagAnalysis_pb2.AnalysisResponse(result = 1,flags = [1],  )




#============================================================
# property
#============================================================
show = ShowVideoStream()



#============================================================
# functions
#============================================================
def serve():


	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
	BagAnalysis_pb2_grpc.add_BagAnalysisServicer_to_server(Greeter(), server)
	server.add_insecure_port('[::]:50051')
	server.start()

	print('===== server start =====')

	try:
		while True:
			time.sleep(0)

	except KeyboardInterrupt:
		server.stop(0)



#============================================================
# main
#============================================================
if __name__ == '__main__':
	show.start()
	serve()


#============================================================
# after the App exit
#============================================================
cv2.destroyAllWindows()