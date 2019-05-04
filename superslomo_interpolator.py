# MIT License
#
# Copyright (c) 2018 Avinash Paliwal
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

import model


class SuperSloMoInterpolator:

    def __init__(self):
        self.device = None
        self.flowBackWarp = None
        self.transform = None
        self.TP = None
        self.flowComp = None
        self.ArbTimeFlowIntrp = None
        self.numThreads = 1


    def Start(self, allowCuda, numThreads):
        torch.cuda.empty_cache()
        self.device = torch.device("cuda" if (torch.cuda.is_available() and allowCuda) else "cpu")
        self.numThreads = numThreads
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


    def UpdateFlowBackWarp(self, fw, fh):
        self.flowBackWarp = model.backWarp(fw, fh, self.device)
        self.flowBackWarp = self.flowBackWarp.to(self.device)


    def Interpolate(self, img1, img2, alpha):
        # reapply the number of threads
        torch.set_num_threads(self.numThreads)

        # resize the ML model to accept our image size if necessary
        if self.flowBackWarp.W != img1.width or self.flowBackWarp.H != img1.height:
            self.UpdateFlowBackWarp(img1.width, img1.height)

        # wrap images around a PyTorch dataloader
        animFrames = AnimationFrameCollection(frame1=img1, frame2=img2, transform=self.transform)
        animFramesloader = torch.utils.data.DataLoader(animFrames)

        with torch.no_grad():
            for (firstFrame, secondFrame) in animFramesloader:
                I0 = firstFrame.to(self.device)
                I1 = secondFrame.to(self.device)

                flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
                F_0_1 = flowOut[:, :2, :, :]
                F_1_0 = flowOut[:, 2:, :, :]

                t = alpha
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                g_I0_F_t_0 = self.flowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = self.flowBackWarp(I1, F_t_1)

                intrpOut = self.ArbTimeFlowIntrp(
                    torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0 = F.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1 = 1 - V_t_0

                g_I0_F_t_0_f = self.flowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = self.flowBackWarp(I1, F_t_1_f)

                wCoeff = [1 - t, t]

                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                            wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
                batchIndex = 0
                interpolatedImg = self.TP(Ft_p[batchIndex].cpu().detach())
                return interpolatedImg


    def Stop(self):
        torch.cuda.empty_cache()
        self.device = None
        self.flowBackWarp = None
        self.transform = None
        self.TP = None
        self.flowComp = None
        self.ArbTimeFlowIntrp = None


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
