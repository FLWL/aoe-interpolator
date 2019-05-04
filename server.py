import time
import grpc
import torch
from PIL import Image
from concurrent import futures
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms


import model
from protos import aoe_interpolator_pb2
from protos import aoe_interpolator_pb2_grpc


class AnimationFrameCollection(data.Dataset):

    def __init__(self, frame1, frame2, transform=None):
        self.frame1 = frame1
        self.frame2 = frame2
        self.transform = transform


    def __getitem__(self, index):
        samples = []

        if self.transform is not None:
            samples.append(self.transform(self.frame1))
            samples.append(self.transform(self.frame2))
        else:
            samples.append(self.frame1)
            samples.append(self.frame2)

        return samples


    def __len__(self):
        return 1


# gRPC listener class
class AoeInterpolatorServicer(aoe_interpolator_pb2_grpc.AoeInterpolatorServicer):

    def __init__(self):
        self.device = None
        self.flowBackWarp = None
        self.transform = None
        self.TP = None
        self.flowComp = None
        self.ArbTimeFlowIntrp = None
        self.numThreads = 1

        print("INIT CALLED!!!!!!!!!!!!!!!!!!!!!!!")


    def StartPyTorch(self, request, context):
        torch.cuda.empty_cache()
        self.device = torch.device("cuda" if (torch.cuda.is_available() and request.allowCuda) else "cpu")
        self.numThreads = request.numThreads
        torch.set_num_threads(self.numThreads)

        mean = [0.429, 0.431, 0.397]
        std = [1, 1, 1]
        normalize = transforms.Normalize(mean=mean, std=std)

        negmean = [x * -1 for x in mean]
        revNormalize = transforms.Normalize(mean=negmean, std=std)

        # Temporary fix for issue #7 https://github.com/avinashpaliwal/Super-SloMo/issues/7 -
        # - Removed per channel mean subtraction for CPU.
        if (self.device == "cpu"):
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.TP = transforms.Compose([transforms.ToPILImage()])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), normalize])
            self.TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

        # Initialize model
        self.flowComp = model.UNet(6, 4)
        self.flowComp.to(self.device)
        for param in self.flowComp.parameters():
            param.requires_grad = False
        self.ArbTimeFlowIntrp = model.UNet(20, 5)
        self.ArbTimeFlowIntrp.to(self.device)
        for param in self.ArbTimeFlowIntrp.parameters():
            param.requires_grad = False

        self.UpdateFlowBackWarp(128, 128)

        checkpoint = "data/SuperSloMo.ckpt"
        dict1 = torch.load(checkpoint, map_location='cpu')
        self.ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
        self.flowComp.load_state_dict(dict1['state_dictFC'])

        return aoe_interpolator_pb2.StartPyTorchResponse(success=True, message='')


    def UpdateFlowBackWarp(self, fw, fh):
        self.flowBackWarp = model.backWarp(fw, fh, self.device)
        self.flowBackWarp = self.flowBackWarp.to(self.device)


    def GetInterpolatedFrame(self, request, context):
        #internal_total_start = time.time()
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

        print("Unified width: " + str(frameWidth))
        print("Unified height: " + str(frameHeight))

        # resize and align both frames into their calculated common size
        img1 = Image.new('RGB', (frameWidth, frameHeight), (255, 0, 255))
        img2 = Image.new('RGB', (frameWidth, frameHeight), (255, 0, 255))
        img1.paste(Image.frombytes('RGB', (frame1Width, frame1Height), request.frame1[16:]),
                   (frame1OffsetX, frame1OffsetY, frame1OffsetX + frame1Width, frame1OffsetY + frame1Height))
        img2.paste(Image.frombytes('RGB', (frame2Width, frame2Height), request.frame2[16:]),
                   (frame2OffsetX, frame2OffsetY, frame2OffsetX + frame2Width, frame2OffsetY + frame2Height))

        # reapply the number of threads
        torch.set_num_threads(self.numThreads)

        # resize the ML model to accept our image size if necessary
        if self.flowBackWarp.W != frameWidth or self.flowBackWarp.H != frameHeight:
            self.UpdateFlowBackWarp(frameWidth, frameHeight)

        # load data
        animFrames = AnimationFrameCollection(frame1=img1, frame2=img2, transform=self.transform)
        animFramesloader = torch.utils.data.DataLoader(animFrames)
        interpolatedImg = None

        #internal_interp_start = time.time()
        with torch.no_grad():
            for (firstFrame, secondFrame) in animFramesloader:
                I0 = firstFrame.to(self.device)
                I1 = secondFrame.to(self.device)

                flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
                F_0_1 = flowOut[:, :2, :, :]
                F_1_0 = flowOut[:, 2:, :, :]

                t = request.alpha
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                g_I0_F_t_0 = self.flowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = self.flowBackWarp(I1, F_t_1)

                intrpOut = self.ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1   = 1 - V_t_0

                g_I0_F_t_0_f = self.flowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = self.flowBackWarp(I1, F_t_1_f)

                wCoeff = [1 - t, t]

                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
                batchIndex = 0
                interpolatedImg = self.TP(Ft_p[batchIndex].cpu().detach())
        #print("Internal interp took: " + str(time.time() - internal_interp_start))

        #img1.save("frame1.png")
        #interpolatedImg.save("frame2.png")
        #img2.save("frame3.png")

        interpolatedBytes = bytearray()
        interpolatedBytes.extend(frameWidth.to_bytes(4, 'little'))
        interpolatedBytes.extend(frameHeight.to_bytes(4, 'little'))
        interpolatedBytes.extend(frameHeight.to_bytes(4, 'little'))
        interpolatedBytes.extend(frameHeight.to_bytes(4, 'little'))
        interpolatedBytes.extend(interpolatedImg.tobytes())
        #print("Internal total took: " + str(time.time() - internal_total_start))
        return aoe_interpolator_pb2.InterpolatedFrameResponse(frame=bytes(interpolatedBytes))


    def roundIntUpToMultipleOf(self, numberToRound, multiple):
        return ((numberToRound + multiple - 1) // multiple) * multiple  # note the integer division


    def StopPyTorch(self, request, context):
        torch.cuda.empty_cache()
        self.device = None
        self.flowBackWarp = None
        self.transform = None
        self.TP = None
        self.flowComp = None
        self.ArbTimeFlowIntrp = None
        return aoe_interpolator_pb2.StopPyTorchResponse(success=True, message='')


if __name__ == '__main__':
    # start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    aoe_interpolator_pb2_grpc.add_AoeInterpolatorServicer_to_server(AoeInterpolatorServicer(), server)
    server.add_insecure_port('[::]:52381')
    server.start()


while True:
    time.sleep(5)
