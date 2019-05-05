# aoe-interpolator

This is a project to provide a 3rd party rendering client for Age of Empires 2 known as CaptureAge with the oppurtunity to display the game's archaic 6-30 FPS animations at the refresh rates of 60+ hertz comparable to modern games. It works by hosting a RPC server that accepts two frames in a specific RGB format and replies with a single interpolated frame at any time between these two frames, chosen by the client. That in between frame is calculated using motion interpolation, more specifically the Super-SloMo implementation that can be found here: https://github.com/avinashpaliwal/Super-SloMo

The client does not need to worry about the size of the frames, or aligning the center points, or cropping the output frames, as that is done automatically be the interpolator.

The RPC server is not guaranteed to complete the frame interpolation in a specific time. It might take anywhere from tens of milliseconds (using GPU acceleration) to multiple seconds (when relying on the CPU). The client is expected to implement their own caching system.

# Communicating with the interpolator

To request an interpolated frame from the RPC server, first the `server.py` must be started and a gRPC call made to port 52381, which looks like:

`rpc GetInterpolatedFrame(InterpolatedFrameRequest) returns (InterpolatedFrameResponse) { }`

and accepts a `InterpolatedFrameRequest` paremeter as follows:
```message InterpolatedFrameRequest
{
    bytes frame1;
    bytes frame2;
    int32 transparentR;
    int32 transparentG;
    int32 transparentB;
    float alpha;
}
```

where `frame1` and `frame2` are are the two frames that the interpolation will be done between, and `transparentR`, `transparentG`, `transparentB` are integers representing the RGB value that acts as a transparent area in the frames. The knowledge of transparency is used for padding incoming frames and cropping outgoing ones.

The `alpha` parameter decides where the interpolated frame should reside between `frame1` and `frame2`. For example, `alpha` of 0.5 would mean that the resulting frame would have to be exactly between the `frame1` and `frame2`. It can be thought of as linear interpolation (lerp). Likewise, `alpha` of 0.3 means that the resulting frame would only have moved 30% towards `frame2` from `frame1`.

Once a frame has been interpolated, the gRPC responds with the following structure:

```message InterpolatedFrameResponse
{
    bytes frame;
}
```

where `frame` is the interpolated frame in the same format as the inputted frames.

# Controlling the interpolator

Once the gRPC connection has been established, the

`rpc StartInterpolator(StartInterpolatorRequest) returns (StartInterpolatorResponse) { }`

call can be made, which notifies the script's PyTorch ML backend to load the model into memory and set everything ready. The `StartInterpolatorRequest` parameter looks like:

```
message StartInterpolatorRequest
{
    bool allowCuda;
    int32 numThreads;
}
```

where `allowCuda` can be set to True to allow the interpolator to use GPU acceleration via CUDA if available, and numThreads to limit the number of CPU threads the interpolator is allowed to use. Note that according to testing by the author, the `numThreads` variable is best left to 1, as a higher number of threads can actually cause a performance regression in this particular program.

There is also a

`rpc StopInterpolator(StopInterpolatorRequest) returns (StopInterpolatorResponse) { }`

call available, which frees the GPU memory and tries to make most of the PyTorch Python objects available for GC, meaning a call to start the interpolator again would have to be made after using this command, if the client wishes to interpolate any more frames. Note that not all of the RAM or VRAM used by PyTorch can be freed, as some will remain cached until script termination.


To gracefully shut down the RPC server, a

`rpc TerminateScript(TerminateScriptRequest) returns (TerminateScriptResponse) { }`

call can be made, which also automatically calls the `StopInterpolator` function before shutting down.

# Input and output formats

The `frame*` parameters seen up above (also .rgb files in the tests directory) are of following C-like structure, with `int` values using little-endian encoding:

```
struct Frame
{
    int width;
    int height;
    int centerX;
    int centerY;
    char data[width*height*3];
}
```

Every entry of the `data` field is a R, G, B value represented by 3 bytes respectively (hence the x3 of the array size).

# Compiling the ProtoBuf file

The .proto file can be found in the ./protos/ directory. To compile it for the Python language,

`python -m grpc_tools.protoc -I../protos --python_out=. --grpc_python_out=. ../protos/aoe_interpolator.proto`

command can be used in that folder. After the compilation has been done, open `aoe_interpolator_pb2_grpc.py` and change the line

`import aoe_interpolator_pb2 as aoe__interpolator__pb2`

to

`from protos import aoe_interpolator_pb2 as aoe__interpolator__pb2`

to account for the fact that the proto files are in a submodule/directory in the project.

The ProtoBuf file can also be compiled for other languages, making it easy to interface with aoe-interpolator.

# Included tests

In the `./tests/` folder the `test_client.py` file can be found, which requests a specific amount of frames to be interpolated from `./tests/data` and its subdirectories, to test the interpolator.

In the `./tests/tools` folder there are files `ca2rgb.py`, `png2rgb.py` and `rgb2png.py`, that are used to convert between different image formats for testing purposes.

# Dependencies

The module depends on Python and its following dependencies:
* grpcio
* protobuf
* torch
* torchvision

# Performance

The performance of this script is limited by the performance of the Super-SloMo interpolator it uses. On a i7-2600k processor, it takes about 2-3 seconds to interpolate a single frame consisting of 160x288 pixels. With CUDA GPU acceleration, the GTX 1080 Ti can output such a frame every 0.03 seconds.
