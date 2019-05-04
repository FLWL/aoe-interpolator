import time
import grpc
from PIL import Image, ImageChops
from concurrent import futures

from protos import aoe_interpolator_pb2
from protos import aoe_interpolator_pb2_grpc
import superslomo_interpolator


# gRPC listener class
class AoeInterpolatorServicer(aoe_interpolator_pb2_grpc.AoeInterpolatorServicer):

    def __init__(self):
        self.interpolator = superslomo_interpolator.SuperSloMoInterpolator()
        self.running = True


    def StartInterpolator(self, request, context):
        self.interpolator.Start(request.allowCuda, request.numThreads)
        return aoe_interpolator_pb2.StartInterpolatorResponse(success=True, message='')


    def GetInterpolatedFrame(self, request, context):
        # frames' metadata
        frame1Width = int.from_bytes(request.frame1[0:4], 'little')
        frame1Height = int.from_bytes(request.frame1[4:8], 'little')
        frame1CenterX = int.from_bytes(request.frame1[8:12], 'little')
        frame1CenterY = int.from_bytes(request.frame1[12:16], 'little')
        frame2Width = int.from_bytes(request.frame2[0:4], 'little')
        frame2Height = int.from_bytes(request.frame2[4:8], 'little')
        frame2CenterX = int.from_bytes(request.frame2[8:12], 'little')
        frame2CenterY = int.from_bytes(request.frame2[12:16], 'little')

        # horizontal offsets
        frame1OffsetX = 0
        frame2OffsetX = 0

        # get X coordinate difference between center points
        xDiff = frame2CenterX - frame1CenterX
        frame1Right = frame1Width - frame1CenterX
        frame2Right = frame2Width - frame2CenterX
        maxFrameRight = max(frame1Right, frame2Right)

        # move the image with smaller centerX to the right
        if xDiff >= 0:
            # move frame1
            frame1OffsetX = xDiff
            frameWidth = frame1OffsetX + frame1CenterX + maxFrameRight # frame1 offset + width
        else:
            # move frame2, because xDiff is negative
            frame2OffsetX = -xDiff
            frameWidth = frame2OffsetX + frame2CenterX + maxFrameRight # frame2 offset + width

        # vertical offsets
        frame1OffsetY = 0
        frame2OffsetY = 0

        # get Y coordinate difference between center points
        yDiff = frame2CenterY - frame1CenterY
        frame1Down = frame1Height - frame1CenterY
        frame2Down = frame2Height - frame2CenterY
        maxFrameDown = max(frame1Down, frame2Down)

        # move the image with smaller centerY down
        if yDiff >= 0:
            # move frame1
            frame1OffsetY = yDiff
            frameHeight = frame1OffsetY + frame1CenterY +  maxFrameDown # frame1 offset + height
        else:
            # move frame2, because xDiff is negative
            frame2OffsetY = -yDiff
            frameHeight = frame2OffsetY + frame2CenterY + maxFrameDown # frame2 offset + height

        # NN requires image dimensions to be in multiples of 32
        # add 1 extra layer of 32px padding to prevent edge artifacts
        widthPadding = self.roundIntUpToMultipleOf(frameWidth, 32) - frameWidth + 32
        heightPadding = self.roundIntUpToMultipleOf(frameHeight, 32) - frameHeight + 32
        frameWidth += widthPadding
        frameHeight += heightPadding
        frame1OffsetX += widthPadding // 2
        frame2OffsetX += widthPadding // 2
        frame1OffsetY += heightPadding // 2
        frame2OffsetY += heightPadding // 2
        frameCenterX = frame1OffsetX + frame1CenterX
        frameCenterY = frame1OffsetY + frame1CenterY

        # resize and align both frames into their calculated common size
        bg = Image.new('RGB', (frameWidth, frameHeight), (request.transparentR, request.transparentG, request.transparentB))
        img1 = bg.copy()
        img2 = bg.copy()
        img1.paste(Image.frombytes('RGB', (frame1Width, frame1Height), request.frame1[16:]),
                   (frame1OffsetX, frame1OffsetY, frame1OffsetX + frame1Width, frame1OffsetY + frame1Height))
        img2.paste(Image.frombytes('RGB', (frame2Width, frame2Height), request.frame2[16:]),
                   (frame2OffsetX, frame2OffsetY, frame2OffsetX + frame2Width, frame2OffsetY + frame2Height))

        # interpolate
        interpolatedImg = self.interpolator.Interpolate(img1, img2, request.alpha)

        # crop
        cropdiff = ImageChops.difference(interpolatedImg, bg) # turn all transparent-y pixels black
        cropdiff = ImageChops.add(cropdiff, cropdiff, 2.0, -20) # also consider pixels that are very close to transparency
        croparea = cropdiff.getbbox()
        frameCenterX -= croparea[0]
        frameCenterY -= croparea[1]
        interpolatedImg = interpolatedImg.crop(croparea)

        croppedWidth = interpolatedImg.width
        croppedHeight = interpolatedImg.height

        # pack the result into bytes and send back
        interpolatedBytes = bytearray()
        interpolatedBytes.extend(croppedWidth.to_bytes(4, 'little'))
        interpolatedBytes.extend(croppedHeight.to_bytes(4, 'little'))
        interpolatedBytes.extend(frameCenterX.to_bytes(4, 'little'))
        interpolatedBytes.extend(frameCenterY.to_bytes(4, 'little'))
        interpolatedBytes.extend(interpolatedImg.tobytes())

        return aoe_interpolator_pb2.InterpolatedFrameResponse(frame=bytes(interpolatedBytes))


    def roundIntUpToMultipleOf(self, numberToRound, multiple):
        return ((numberToRound + multiple - 1) // multiple) * multiple  # note the integer division


    def StopInterpolator(self, request, context):
        self.interpolator.Stop()
        return aoe_interpolator_pb2.StopInterpolatorResponse(success=True, message='')


    def TerminateScript(self, request, context):
        if self.interpolator.device:
            self.interpolator.Stop()

        self.running = False
        return aoe_interpolator_pb2.TerminateScriptResponse(success=True, message='')


if __name__ == '__main__':
    # start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    servicer = AoeInterpolatorServicer()
    aoe_interpolator_pb2_grpc.add_AoeInterpolatorServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:52381')
    server.start()
    print("Started gRPC server...")

    while servicer.running:
        time.sleep(0.5)

    time.sleep(0.5)
    server.stop(True)
    print("Stopped gRPC server...")
