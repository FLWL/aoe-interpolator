# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: aoe_interpolator.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='aoe_interpolator.proto',
  package='aoeinterpolator',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x16\x61oe_interpolator.proto\x12\x0f\x61oeinterpolator\"A\n\x18StartInterpolatorRequest\x12\x11\n\tallowCuda\x18\x01 \x01(\x08\x12\x12\n\nnumThreads\x18\x02 \x01(\x05\"=\n\x19StartInterpolatorResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\"\x8b\x01\n\x18InterpolatedFrameRequest\x12\x0e\n\x06\x66rame1\x18\x01 \x01(\x0c\x12\x0e\n\x06\x66rame2\x18\x02 \x01(\x0c\x12\x14\n\x0ctransparentR\x18\x03 \x01(\x05\x12\x14\n\x0ctransparentG\x18\x04 \x01(\x05\x12\x14\n\x0ctransparentB\x18\x05 \x01(\x05\x12\r\n\x05\x61lpha\x18\x06 \x01(\x02\"*\n\x19InterpolatedFrameResponse\x12\r\n\x05\x66rame\x18\x01 \x01(\x0c\"\x19\n\x17StopInterpolatorRequest\"<\n\x18StopInterpolatorResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\"\x18\n\x16TerminateScriptRequest\";\n\x17TerminateScriptResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t2\xc3\x03\n\x0f\x41oeInterpolator\x12l\n\x11StartInterpolator\x12).aoeinterpolator.StartInterpolatorRequest\x1a*.aoeinterpolator.StartInterpolatorResponse\"\x00\x12o\n\x14GetInterpolatedFrame\x12).aoeinterpolator.InterpolatedFrameRequest\x1a*.aoeinterpolator.InterpolatedFrameResponse\"\x00\x12i\n\x10StopInterpolator\x12(.aoeinterpolator.StopInterpolatorRequest\x1a).aoeinterpolator.StopInterpolatorResponse\"\x00\x12\x66\n\x0fTerminateScript\x12\'.aoeinterpolator.TerminateScriptRequest\x1a(.aoeinterpolator.TerminateScriptResponse\"\x00\x62\x06proto3')
)




_STARTINTERPOLATORREQUEST = _descriptor.Descriptor(
  name='StartInterpolatorRequest',
  full_name='aoeinterpolator.StartInterpolatorRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='allowCuda', full_name='aoeinterpolator.StartInterpolatorRequest.allowCuda', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='numThreads', full_name='aoeinterpolator.StartInterpolatorRequest.numThreads', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=43,
  serialized_end=108,
)


_STARTINTERPOLATORRESPONSE = _descriptor.Descriptor(
  name='StartInterpolatorResponse',
  full_name='aoeinterpolator.StartInterpolatorResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='aoeinterpolator.StartInterpolatorResponse.success', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message', full_name='aoeinterpolator.StartInterpolatorResponse.message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=110,
  serialized_end=171,
)


_INTERPOLATEDFRAMEREQUEST = _descriptor.Descriptor(
  name='InterpolatedFrameRequest',
  full_name='aoeinterpolator.InterpolatedFrameRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='frame1', full_name='aoeinterpolator.InterpolatedFrameRequest.frame1', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frame2', full_name='aoeinterpolator.InterpolatedFrameRequest.frame2', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='transparentR', full_name='aoeinterpolator.InterpolatedFrameRequest.transparentR', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='transparentG', full_name='aoeinterpolator.InterpolatedFrameRequest.transparentG', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='transparentB', full_name='aoeinterpolator.InterpolatedFrameRequest.transparentB', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='alpha', full_name='aoeinterpolator.InterpolatedFrameRequest.alpha', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=174,
  serialized_end=313,
)


_INTERPOLATEDFRAMERESPONSE = _descriptor.Descriptor(
  name='InterpolatedFrameResponse',
  full_name='aoeinterpolator.InterpolatedFrameResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='frame', full_name='aoeinterpolator.InterpolatedFrameResponse.frame', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=315,
  serialized_end=357,
)


_STOPINTERPOLATORREQUEST = _descriptor.Descriptor(
  name='StopInterpolatorRequest',
  full_name='aoeinterpolator.StopInterpolatorRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=359,
  serialized_end=384,
)


_STOPINTERPOLATORRESPONSE = _descriptor.Descriptor(
  name='StopInterpolatorResponse',
  full_name='aoeinterpolator.StopInterpolatorResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='aoeinterpolator.StopInterpolatorResponse.success', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message', full_name='aoeinterpolator.StopInterpolatorResponse.message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=386,
  serialized_end=446,
)


_TERMINATESCRIPTREQUEST = _descriptor.Descriptor(
  name='TerminateScriptRequest',
  full_name='aoeinterpolator.TerminateScriptRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=448,
  serialized_end=472,
)


_TERMINATESCRIPTRESPONSE = _descriptor.Descriptor(
  name='TerminateScriptResponse',
  full_name='aoeinterpolator.TerminateScriptResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='aoeinterpolator.TerminateScriptResponse.success', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message', full_name='aoeinterpolator.TerminateScriptResponse.message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=474,
  serialized_end=533,
)

DESCRIPTOR.message_types_by_name['StartInterpolatorRequest'] = _STARTINTERPOLATORREQUEST
DESCRIPTOR.message_types_by_name['StartInterpolatorResponse'] = _STARTINTERPOLATORRESPONSE
DESCRIPTOR.message_types_by_name['InterpolatedFrameRequest'] = _INTERPOLATEDFRAMEREQUEST
DESCRIPTOR.message_types_by_name['InterpolatedFrameResponse'] = _INTERPOLATEDFRAMERESPONSE
DESCRIPTOR.message_types_by_name['StopInterpolatorRequest'] = _STOPINTERPOLATORREQUEST
DESCRIPTOR.message_types_by_name['StopInterpolatorResponse'] = _STOPINTERPOLATORRESPONSE
DESCRIPTOR.message_types_by_name['TerminateScriptRequest'] = _TERMINATESCRIPTREQUEST
DESCRIPTOR.message_types_by_name['TerminateScriptResponse'] = _TERMINATESCRIPTRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

StartInterpolatorRequest = _reflection.GeneratedProtocolMessageType('StartInterpolatorRequest', (_message.Message,), dict(
  DESCRIPTOR = _STARTINTERPOLATORREQUEST,
  __module__ = 'aoe_interpolator_pb2'
  # @@protoc_insertion_point(class_scope:aoeinterpolator.StartInterpolatorRequest)
  ))
_sym_db.RegisterMessage(StartInterpolatorRequest)

StartInterpolatorResponse = _reflection.GeneratedProtocolMessageType('StartInterpolatorResponse', (_message.Message,), dict(
  DESCRIPTOR = _STARTINTERPOLATORRESPONSE,
  __module__ = 'aoe_interpolator_pb2'
  # @@protoc_insertion_point(class_scope:aoeinterpolator.StartInterpolatorResponse)
  ))
_sym_db.RegisterMessage(StartInterpolatorResponse)

InterpolatedFrameRequest = _reflection.GeneratedProtocolMessageType('InterpolatedFrameRequest', (_message.Message,), dict(
  DESCRIPTOR = _INTERPOLATEDFRAMEREQUEST,
  __module__ = 'aoe_interpolator_pb2'
  # @@protoc_insertion_point(class_scope:aoeinterpolator.InterpolatedFrameRequest)
  ))
_sym_db.RegisterMessage(InterpolatedFrameRequest)

InterpolatedFrameResponse = _reflection.GeneratedProtocolMessageType('InterpolatedFrameResponse', (_message.Message,), dict(
  DESCRIPTOR = _INTERPOLATEDFRAMERESPONSE,
  __module__ = 'aoe_interpolator_pb2'
  # @@protoc_insertion_point(class_scope:aoeinterpolator.InterpolatedFrameResponse)
  ))
_sym_db.RegisterMessage(InterpolatedFrameResponse)

StopInterpolatorRequest = _reflection.GeneratedProtocolMessageType('StopInterpolatorRequest', (_message.Message,), dict(
  DESCRIPTOR = _STOPINTERPOLATORREQUEST,
  __module__ = 'aoe_interpolator_pb2'
  # @@protoc_insertion_point(class_scope:aoeinterpolator.StopInterpolatorRequest)
  ))
_sym_db.RegisterMessage(StopInterpolatorRequest)

StopInterpolatorResponse = _reflection.GeneratedProtocolMessageType('StopInterpolatorResponse', (_message.Message,), dict(
  DESCRIPTOR = _STOPINTERPOLATORRESPONSE,
  __module__ = 'aoe_interpolator_pb2'
  # @@protoc_insertion_point(class_scope:aoeinterpolator.StopInterpolatorResponse)
  ))
_sym_db.RegisterMessage(StopInterpolatorResponse)

TerminateScriptRequest = _reflection.GeneratedProtocolMessageType('TerminateScriptRequest', (_message.Message,), dict(
  DESCRIPTOR = _TERMINATESCRIPTREQUEST,
  __module__ = 'aoe_interpolator_pb2'
  # @@protoc_insertion_point(class_scope:aoeinterpolator.TerminateScriptRequest)
  ))
_sym_db.RegisterMessage(TerminateScriptRequest)

TerminateScriptResponse = _reflection.GeneratedProtocolMessageType('TerminateScriptResponse', (_message.Message,), dict(
  DESCRIPTOR = _TERMINATESCRIPTRESPONSE,
  __module__ = 'aoe_interpolator_pb2'
  # @@protoc_insertion_point(class_scope:aoeinterpolator.TerminateScriptResponse)
  ))
_sym_db.RegisterMessage(TerminateScriptResponse)



_AOEINTERPOLATOR = _descriptor.ServiceDescriptor(
  name='AoeInterpolator',
  full_name='aoeinterpolator.AoeInterpolator',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=536,
  serialized_end=987,
  methods=[
  _descriptor.MethodDescriptor(
    name='StartInterpolator',
    full_name='aoeinterpolator.AoeInterpolator.StartInterpolator',
    index=0,
    containing_service=None,
    input_type=_STARTINTERPOLATORREQUEST,
    output_type=_STARTINTERPOLATORRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetInterpolatedFrame',
    full_name='aoeinterpolator.AoeInterpolator.GetInterpolatedFrame',
    index=1,
    containing_service=None,
    input_type=_INTERPOLATEDFRAMEREQUEST,
    output_type=_INTERPOLATEDFRAMERESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='StopInterpolator',
    full_name='aoeinterpolator.AoeInterpolator.StopInterpolator',
    index=2,
    containing_service=None,
    input_type=_STOPINTERPOLATORREQUEST,
    output_type=_STOPINTERPOLATORRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='TerminateScript',
    full_name='aoeinterpolator.AoeInterpolator.TerminateScript',
    index=3,
    containing_service=None,
    input_type=_TERMINATESCRIPTREQUEST,
    output_type=_TERMINATESCRIPTRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_AOEINTERPOLATOR)

DESCRIPTOR.services_by_name['AoeInterpolator'] = _AOEINTERPOLATOR

# @@protoc_insertion_point(module_scope)
