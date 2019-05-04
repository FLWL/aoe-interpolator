import grpc
import time

from protos import aoe_interpolator_pb2
from protos import aoe_interpolator_pb2_grpc
from tests.tools import rgb2png
from tests.tools import png2rgb


def run():
    # input parameters
    frame1Path = "data/input_bytes/frame1.rgb"
    frame2Path = "data/input_bytes/frame2.rgb"
    numExtraFrames = 10

    # also possible to convert png images to rgb
    # although this assumes that the centerX and centerY are in the middle
    #png2rgb.png2rgb("data/input_bytes/frame1.png", frame1Path)
    #png2rgb.png2rgb("data/input_bytes/frame2.png", frame2Path)

    # debug inputs as png files
    rgb2png.rgb2png(frame1Path, "data/input_pngs/frame1.png")
    rgb2png.rgb2png(frame2Path, "data/input_pngs/frame2.png")

    with grpc.insecure_channel('localhost:52381') as channel:
        stub = aoe_interpolator_pb2_grpc.AoeInterpolatorStub(channel)

        # load frames
        with open(frame1Path, "rb") as frame1File: frame1 = frame1File.read()
        with open(frame2Path, "rb") as frame2File: frame2 = frame2File.read()

        # init interpolator
        init_start = time.time()
        stub.StartInterpolator(aoe_interpolator_pb2.StartInterpolatorRequest(allowCuda=True, numThreads=1))
        print("AoeInterpolator initialization took: " + str(time.time() - init_start))

        frametimes = []
        frametimes.append(0.0)

        timestep = 1.0 / (numExtraFrames + 1)
        cur_time = 0.0
        for i in range(numExtraFrames):
            cur_time += timestep
            frametimes.append(cur_time)

        frametimes.append(1.0)
        for i, frametime in enumerate(frametimes):
            frame_num = i + 1
            frametime_formatted = "{0:.2f}".format(frametime)

            # call to interpolate a frame
            interp_start = time.time()
            interpolationRequest = aoe_interpolator_pb2.InterpolatedFrameRequest()
            interpolationRequest.frame1 = frame1
            interpolationRequest.frame2 = frame2
            interpolationRequest.transparentR = 255
            interpolationRequest.transparentG = 0
            interpolationRequest.transparentB = 255
            interpolationRequest.alpha = float(frametime)
            interpolatedFrameResponse = stub.GetInterpolatedFrame(interpolationRequest)
            print("Frame " + str(frame_num) + " (" + frametime_formatted + ") interpolation took: " + str(time.time() - interp_start))

            # save the output frame
            framePath = "data/output_bytes/frame" + str(frame_num) + "_(" + frametime_formatted +").rgb"
            with open(framePath, "wb") as f:
                f.write(interpolatedFrameResponse.frame)

            # debug png
            rgb2png.rgb2png(framePath, "data/output_pngs/frame" + str(frame_num) + "_(" + frametime_formatted +").png")

        # tell the server to stop
        stub.StopInterpolator(aoe_interpolator_pb2.StopInterpolatorRequest())
        stub.TerminateScript(aoe_interpolator_pb2.TerminateScriptRequest())

if __name__ == '__main__':
    run()
