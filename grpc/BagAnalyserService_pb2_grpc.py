# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import BagAnalyserService_pb2 as BagAnalyserService__pb2


class BagAnalysisServiceStub(object):
  """Interface exported by the server.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.BeginSession = channel.unary_unary(
        '/Sym3.BagAnalysis.BagAnalysisService/BeginSession',
        request_serializer=BagAnalyserService__pb2.BeginSessionRequest.SerializeToString,
        response_deserializer=BagAnalyserService__pb2.BeginSessionResponse.FromString,
        )
    self.EndSession = channel.unary_unary(
        '/Sym3.BagAnalysis.BagAnalysisService/EndSession',
        request_serializer=BagAnalyserService__pb2.ClientIdentifier.SerializeToString,
        response_deserializer=BagAnalyserService__pb2.EndSessionResponse.FromString,
        )
    self.StartAnalyse = channel.unary_stream(
        '/Sym3.BagAnalysis.BagAnalysisService/StartAnalyse',
        request_serializer=BagAnalyserService__pb2.AnalyseRequest.SerializeToString,
        response_deserializer=BagAnalyserService__pb2.AnalyseResponse.FromString,
        )
    self.StopAnalyse = channel.unary_unary(
        '/Sym3.BagAnalysis.BagAnalysisService/StopAnalyse',
        request_serializer=BagAnalyserService__pb2.ClientIdentifier.SerializeToString,
        response_deserializer=BagAnalyserService__pb2.StopAnalyseResponse.FromString,
        )
    self.Analyse = channel.unary_unary(
        '/Sym3.BagAnalysis.BagAnalysisService/Analyse',
        request_serializer=BagAnalyserService__pb2.AnalyseRequest.SerializeToString,
        response_deserializer=BagAnalyserService__pb2.AnalyseResponse.FromString,
        )
    self.Ping = channel.unary_unary(
        '/Sym3.BagAnalysis.BagAnalysisService/Ping',
        request_serializer=BagAnalyserService__pb2.ClientIdentifier.SerializeToString,
        response_deserializer=BagAnalyserService__pb2.PingResponse.FromString,
        )


class BagAnalysisServiceServicer(object):
  """Interface exported by the server.
  """

  def BeginSession(self, request, context):
    """Starts a new analysis session between client and server.


    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def EndSession(self, request, context):
    """Ends current session
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def StartAnalyse(self, request, context):
    """Starts 
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def StopAnalyse(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Analyse(self, request, context):
    """Performs a single analysis
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Ping(self, request, context):
    """Ping function
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_BagAnalysisServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'BeginSession': grpc.unary_unary_rpc_method_handler(
          servicer.BeginSession,
          request_deserializer=BagAnalyserService__pb2.BeginSessionRequest.FromString,
          response_serializer=BagAnalyserService__pb2.BeginSessionResponse.SerializeToString,
      ),
      'EndSession': grpc.unary_unary_rpc_method_handler(
          servicer.EndSession,
          request_deserializer=BagAnalyserService__pb2.ClientIdentifier.FromString,
          response_serializer=BagAnalyserService__pb2.EndSessionResponse.SerializeToString,
      ),
      'StartAnalyse': grpc.unary_stream_rpc_method_handler(
          servicer.StartAnalyse,
          request_deserializer=BagAnalyserService__pb2.AnalyseRequest.FromString,
          response_serializer=BagAnalyserService__pb2.AnalyseResponse.SerializeToString,
      ),
      'StopAnalyse': grpc.unary_unary_rpc_method_handler(
          servicer.StopAnalyse,
          request_deserializer=BagAnalyserService__pb2.ClientIdentifier.FromString,
          response_serializer=BagAnalyserService__pb2.StopAnalyseResponse.SerializeToString,
      ),
      'Analyse': grpc.unary_unary_rpc_method_handler(
          servicer.Analyse,
          request_deserializer=BagAnalyserService__pb2.AnalyseRequest.FromString,
          response_serializer=BagAnalyserService__pb2.AnalyseResponse.SerializeToString,
      ),
      'Ping': grpc.unary_unary_rpc_method_handler(
          servicer.Ping,
          request_deserializer=BagAnalyserService__pb2.ClientIdentifier.FromString,
          response_serializer=BagAnalyserService__pb2.PingResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'Sym3.BagAnalysis.BagAnalysisService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))