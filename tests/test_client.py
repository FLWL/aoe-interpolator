import grpc
import time

from protos import aoe_interpolator_pb2
from protos import aoe_interpolator_pb2_grpc

frame1Path = "data/facet_0_frame_0.rgb"
frame2Path = "data/facet_0_frame_1.rgb"
outputPath = "data/output.rgb"

def run():
    with grpc.insecure_channel('localhost:52381') as channel:
        stub = aoe_interpolator_pb2_grpc.AoeInterpolatorStub(channel)

        # load frames
        with open(frame1Path, "rb") as frame1File: frame1 = frame1File.read()
        with open(frame2Path, "rb") as frame2File: frame2 = frame2File.read()

        init_start = time.time()
        stub.StartPyTorch(aoe_interpolator_pb2.StartPyTorchRequest(allowCuda=True, numThreads=1))
        print("Init took: " + str(time.time() - init_start))

        total_time = 0.0
        amt = 500
        for i in range(amt):
            interp_start = time.time()
            response = stub.GetInterpolatedFrame(
                aoe_interpolator_pb2.InterpolatedFrameRequest(frame1=frame1, frame2=frame2, alpha=float(0.5)))
            interp_took = time.time() - interp_start
            print("Interp took: " + str(interp_took))
            total_time += interp_took


        avg = total_time / amt
        print("Average time: " + str(avg))

    print("Test client received: " + str(response.frame))
    with open(outputPath, "wb") as f:
        f.write(response.frame)


if __name__ == '__main__':
    run()
