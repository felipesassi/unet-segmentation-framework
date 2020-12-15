import torch
import torch.nn as nn
from torch.nn import functional as F

import math

import numpy as np

torch.manual_seed(0)

"""
This code is based in this article - U-Net: Convolutional Networks for Biomedical Image Segmentation - https://arxiv.org/pdf/1505.04597.pdf
"""


class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, init_depth):
        super().__init__()
        assert in_channels >= 1, "in_channels must be greater than 0"
        assert out_channels >= 1, "out_channels must be greater than 0"
        assert init_depth >= 1, "init_depth must be greater than 0"
        # Encoder part
        self.Encoder_1 = self.layers_block(in_channels, init_depth)
        self.Encoder_MaxPool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Encoder_2 = self.layers_block(1*init_depth, 2*init_depth)
        self.Encoder_MaxPool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Encoder_3 = self.layers_block(2*init_depth, 4*init_depth)
        self.Encoder_MaxPool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Encoder_4 = self.layers_block(4*init_depth, 8*init_depth)
        self.Encoder_MaxPool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bottleneck part
        self.Encoder_5 = self.layers_block(
            8*init_depth, 16*init_depth)  # bottleneck
        # Decoder part
        self.Decoder_1 = self.layers_block(16*init_depth, 8*init_depth)
        self.Decoder_UpSampling_1 = nn.ConvTranspose2d(
            in_channels=16*init_depth, out_channels=8*init_depth, kernel_size=2, stride=2)
        self.Decoder_2 = self.layers_block(8*init_depth, 4*init_depth)
        self.Decoder_UpSampling_2 = nn.ConvTranspose2d(
            in_channels=8*init_depth, out_channels=4*init_depth, kernel_size=2, stride=2)
        self.Decoder_3 = self.layers_block(4*init_depth, 2*init_depth)
        self.Decoder_UpSampling_3 = nn.ConvTranspose2d(
            in_channels=4*init_depth, out_channels=2*init_depth, kernel_size=2, stride=2)
        self.Decoder_4 = self.layers_block(2*init_depth, init_depth)
        self.Decoder_UpSampling_4 = nn.ConvTranspose2d(
            in_channels=2*init_depth, out_channels=1*init_depth, kernel_size=2, stride=2)
        # Output
        self.Output = nn.Conv2d(
            in_channels=init_depth, out_channels=out_channels, kernel_size=1, stride=1)

    @staticmethod
    def layers_block(in_channels, out_channels):
        """
        This static method create a convolutional block useful for the U-Net implementation
        """
        return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm2d(num_features=out_channels),
                             nn.ReLU(),
                             nn.Conv2d(
                                 in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm2d(num_features=out_channels),
                             nn.ReLU()
                             )

    @staticmethod
    def calculate_pad(x_1, x_2):
        padding_x = np.uint8(x_1.shape[2] - x_2.shape[2])
        padding_y = np.uint8(x_1.shape[3] - x_2.shape[3])
        first_dim_x = math.floor(padding_x/2)
        second_dim_x = padding_x - first_dim_x
        first_dim_y = math.floor(padding_y/2)
        second_dim_y = padding_y - first_dim_y
        return F.pad(x_2, (first_dim_y, second_dim_y, first_dim_x, second_dim_x))

    def forward(self, x):
        #assert x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0, "The image input dimensions must be divisible by 2"
        encoder_1 = self.Encoder_1(x)
        encoder_2 = self.Encoder_2(self.Encoder_MaxPool_1(encoder_1))
        encoder_3 = self.Encoder_3(self.Encoder_MaxPool_2(encoder_2))
        encoder_4 = self.Encoder_4(self.Encoder_MaxPool_3(encoder_3))
        encoder_5 = self.Encoder_5(self.Encoder_MaxPool_4(encoder_4))

        #decoder_1 = self.Decoder_1(encoder_4, encoder_5)
        #decoder_2 = self.Decoder_2(encoder_3, decoder_1)
        #decoder_3 = self.Decoder_3(encoder_2, decoder_2)
        #decoder_4 = self.Decoder_4(encoder_1, decoder_3)

        encoder_5 = self.calculate_pad(
            encoder_4, self.Decoder_UpSampling_1(encoder_5))
        decoder_1 = self.Decoder_1(torch.cat([encoder_4, encoder_5], dim=1))

        decoder_1 = self.calculate_pad(
            encoder_3, self.Decoder_UpSampling_2(decoder_1))
        decoder_2 = self.Decoder_2(torch.cat([encoder_3, decoder_1], dim=1))

        decoder_2 = self.calculate_pad(
            encoder_2, self.Decoder_UpSampling_3(decoder_2))
        decoder_3 = self.Decoder_3(torch.cat([encoder_2, decoder_2], dim=1))

        decoder_3 = self.calculate_pad(
            encoder_1, self.Decoder_UpSampling_4(decoder_3))
        decoder_4 = self.Decoder_4(torch.cat([encoder_1, decoder_3], dim=1))

        #decoder_1 = self.Decoder_1(torch.cat([encoder_4, self.Decoder_UpSampling_1(encoder_5)], dim = 1))

        # print(decoder_1)

        #decoder_2 = self.Decoder_2(torch.cat([encoder_3, self.Decoder_UpSampling_2(decoder_1)], dim = 1))
        #decoder_3 = self.Decoder_3(torch.cat([encoder_2, self.Decoder_UpSampling_3(decoder_2)], dim = 1))

        #decoder_4 = self.Decoder_4(torch.cat([encoder_1, self.Decoder_UpSampling_4(decoder_3)], dim = 1))
        output = self.Output(decoder_4)
        return output


def foo(teste=1):
    print('sd')


if __name__ == '__main__':
    teste = UNet2D(1, 1, 2)
    tensor = torch.zeros((1, 1, 120, 120))
    print(teste(tensor).shape)
