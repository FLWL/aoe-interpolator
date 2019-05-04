# aoe-interpolator

This is a project to provide a 3rd party rendering client for Age of Empires 2 known as CaptureAge with the oppurtunity to display the game's archaic 6-30 FPS animations at the refresh rates of 60+ hertz comparable to modern games. It works by hosting a RPC server that accepts two frames in a specific RGB format and replies with a single interpolated frame at any time between these two frames, chosen by the client. That in between frame is calculated using motion interpolation, more specifically the Super-SloMo implementation that can be found here: https://github.com/avinashpaliwal/Super-SloMo

The RPC server is not guaranteed to complete the frame interpolation in a specific time. It might take anywhere from tens of milliseconds (using GPU acceleration) to multiple seconds (when relying on the CPU). The client is expected to implement their own caching system.

# Communicating with the interpolator

To request an interpolated frame from the RPC server, first the `server.py` must be started and a gRPC call made, which looks like:

`rpc GetInterpolatedFrame(InterpolatedFrameRequest) returns (InterpolatedFrameResponse) { }`

and accepts a `InterpolatedFrameRequest` paremeter as follows:
```message InterpolatedFrameRequest
{
    bytes frame1 ;
    bytes frame2;
    int32 transparentR;
    int32 transparentG;
    int32 transparentB;
    float alpha;
}
```

where `frame1` and `frame2` are are the two frames that the interpolation will be done between, and `transparentR`, `transparentG`, `transparentB` are integers representing the RGB value that acts as a transparent area in the frames. The knowledge of transparency is used for padding incoming frames and cropping outgoing ones.

The `alpha` parameter decides where the interpolated frame should reside between `frame1` and `frame2`. For example, `alpha` of 0.5 would mean that the resulting frame would have to be exactly between the `frame1` and `frame2`. It can be thought of as linear interpolation (lerp). Likewise, `alpha` of 0.3 means that the resulting frame would only have moved 30% towards `frame2` from `frame1`.

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

call available, which frees the GPU memory and tries to make most of the PyTorch Python objects free for GC, meaning a call to start the interpolator again would have to be made after using this command, if the client wishes to interpolate any more frames. Note that not 100% RAM or VRAM can be freed, some will remain cached to PyTorch until script termination.


To gracefully shut down the RPC server, a

`rpc TerminateScript(TerminateScriptRequest) returns (TerminateScriptResponse) { }`

call can be made, which also automatically calls the `StopInterpolator` function before shutting down.




