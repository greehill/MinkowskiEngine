# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
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
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.

import unittest
import numpy as np

import torch
import torch.nn as nn

import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor
from MinkowskiEngine.utils import summary, batched_coordinates
from tests.python.common import load_file


class StackUNet(ME.MinkowskiNetwork):
    def __init__(self, in_nchannel, out_nchannel, D):
        ME.MinkowskiNetwork.__init__(self, D)
        channels = [in_nchannel, 16, 32]
        self.net = nn.Sequential(
            ME.MinkowskiStackSum(
                ME.MinkowskiConvolution(
                    channels[0],
                    channels[1],
                    kernel_size=3,
                    stride=1,
                    dimension=D,
                ),
                nn.Sequential(
                    ME.MinkowskiConvolution(
                        channels[0],
                        channels[1],
                        kernel_size=3,
                        stride=2,
                        dimension=D,
                    ),
                    ME.MinkowskiStackSum(
                        nn.Identity(),
                        nn.Sequential(
                            ME.MinkowskiConvolution(
                                channels[1],
                                channels[2],
                                kernel_size=3,
                                stride=2,
                                dimension=D,
                            ),
                            ME.MinkowskiConvolutionTranspose(
                                channels[2],
                                channels[1],
                                kernel_size=3,
                                stride=1,
                                dimension=D,
                            ),
                            ME.MinkowskiPoolingTranspose(
                                kernel_size=2, stride=2, dimension=D
                            ),
                        ),
                    ),
                    ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=D),
                ),
            ),
            ME.MinkowskiToFeature(),
            nn.Linear(channels[1], out_nchannel, bias=True),
        )

    def forward(self, x):
        return self.net(x)



class TestSummary(unittest.TestCase):
    def setUp(self):
        voxel_size = 0.02
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = StackUNet(3, 20, D=3).to(self.device)
        coords, colors, _ = load_file("1.ply")

        self.sinput = SparseTensor(
            features=torch.from_numpy(colors).float(),
            coordinates=batched_coordinates([coords / voxel_size], dtype=torch.float32),
            device=self.device,
        )

    def test(self):
        summary(self.net, self.sinput)
