# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from protos import aoe_interpolator_pb2 as aoe__interpolator__pb2


class AoeInterpolatorStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.StartPyTorch = channel.unary_unary(
        '/aoeinterpolator.AoeInterpolator/StartPyTorch',
        request_serializer=aoe__interpolator__pb2.StartPyTorchRequest.SerializeToString,
        response_deserializer=aoe__interpolator__pb2.StartPyTorchResponse.FromString,
        )
    self.GetInterpolatedFrame = channel.unary_unary(
        '/aoeinterpolator.AoeInterpolator/GetInterpolatedFrame',
        request_serializer=aoe__interpolator__pb2.InterpolatedFrameRequest.SerializeToString,
        response_deserializer=aoe__interpolator__pb2.InterpolatedFrameResponse.FromString,
        )
    self.StopPyTorch = channel.unary_unary(
        '/aoeinterpolator.AoeInterpolator/StopPyTorch',
        request_serializer=aoe__interpolator__pb2.StopPyTorchRequest.SerializeToString,
        response_deserializer=aoe__interpolator__pb2.StopPyTorchResponse.FromString,
        )


class AoeInterpolatorServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def StartPyTorch(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetInterpolatedFrame(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def StopPyTorch(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_AoeInterpolatorServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'StartPyTorch': grpc.unary_unary_rpc_method_handler(
          servicer.StartPyTorch,
          request_deserializer=aoe__interpolator__pb2.StartPyTorchRequest.FromString,
          response_serializer=aoe__interpolator__pb2.StartPyTorchResponse.SerializeToString,
      ),
      'GetInterpolatedFrame': grpc.unary_unary_rpc_method_handler(
          servicer.GetInterpolatedFrame,
          request_deserializer=aoe__interpolator__pb2.InterpolatedFrameRequest.FromString,
          response_serializer=aoe__interpolator__pb2.InterpolatedFrameResponse.SerializeToString,
      ),
      'StopPyTorch': grpc.unary_unary_rpc_method_handler(
          servicer.StopPyTorch,
          request_deserializer=aoe__interpolator__pb2.StopPyTorchRequest.FromString,
          response_serializer=aoe__interpolator__pb2.StopPyTorchResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'aoeinterpolator.AoeInterpolator', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
