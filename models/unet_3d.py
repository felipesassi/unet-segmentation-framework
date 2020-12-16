import torch
import torch.nn as nn
from torch.nn import functional as F

import math

import numpy as np

torch.manual_seed(0)

"""
This code is based in this article - U-Net: Convolutional Networks for Biomedical Image Segmentation - https://arxiv.org/pdf/1505.04597.pdf
"""


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, init_depth):
        super().__init__()
        assert in_channels >= 1, "in_channels must be greater than 0"
        assert out_channels >= 1, "out_channels must be greater than 0"
        assert init_depth >= 1, "init_depth must be greater than 0"
        # Encoder part
        self.encoder_1 = self.layers_block(in_channels, init_depth)
        self.encoder_maxpool_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder_2 = self.layers_block(1*init_depth, 2*init_depth)
        self.encoder_maxpool_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder_3 = self.layers_block(2*init_depth, 4*init_depth)
        self.encoder_maxpool_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder_4 = self.layers_block(4*init_depth, 8*init_depth)
        self.encoder_maxpool_4 = nn.MaxPool3d(kernel_size=2, stride=2)
        # Bottleneck part
        self.encoder_5 = self.layers_block(
            8*init_depth, 16*init_depth)  # bottleneck
        # Decoder part
        self.decoder_1 = self.layers_block(16*init_depth, 8*init_depth)
        self.decoder_upsampling_1 = nn.ConvTranspose3d(
            in_channels=16*init_depth, out_channels=8*init_depth, kernel_size=2, stride=2, padding=0)
        self.decoder_2 = self.layers_block(8*init_depth, 4*init_depth)
        self.decoder_upsampling_2 = nn.ConvTranspose3d(
            in_channels=8*init_depth, out_channels=4*init_depth, kernel_size=2, stride=2, padding=0)
        self.decoder_3 = self.layers_block(4*init_depth, 2*init_depth)
        self.decoder_upsampling_3 = nn.ConvTranspose3d(
            in_channels=4*init_depth, out_channels=2*init_depth, kernel_size=2, stride=2, padding=0)
        self.decoder_4 = self.layers_block(2*init_depth, init_depth)
        self.decoder_upsampling_4 = nn.ConvTranspose3d(
            in_channels=2*init_depth, out_channels=1*init_depth, kernel_size=2, stride=2, padding=0)
        # Output
        self.output = nn.Conv3d(
            in_channels=init_depth, out_channels=out_channels, kernel_size=1, stride=1)

    @staticmethod
    def layers_block(in_channels, out_channels):
        """
        This static method create a convolutional block useful for the U-Net implementation
        """
        return nn.Sequential(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                            
                             nn.InstanceNorm3d(num_features=out_channels),
                             nn.ReLU(),
                             nn.Conv3d(
                                 in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                             nn.InstanceNorm3d(num_features=out_channels),
                             nn.ReLU()
                             )

    @staticmethod
    def calculate_pad(x_1, x_2):
        padding_x = np.uint8(x_1.shape[2] - x_2.shape[2])
        padding_y = np.uint8(x_1.shape[3] - x_2.shape[3])
        padding_z = np.uint8(x_1.shape[4] - x_2.shape[4])
        first_dim_x = math.floor(padding_x/2)
        second_dim_x = padding_x - first_dim_x
        first_dim_y = math.floor(padding_y/2)
        second_dim_y = padding_y - first_dim_y
        first_dim_z = math.floor(padding_z/2)
        second_dim_z = padding_z - first_dim_z
        return F.pad(x_2, (first_dim_y, second_dim_y, first_dim_x, second_dim_x, second_dim_z, second_dim_z))

    def forward(self, x):
        #assert x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0, "The image input dimensions must be divisible by 2"
        encoder_1 = self.encoder_1(x)
        encoder_2 = self.encoder_2(self.encoder_maxpool_1(encoder_1))
        encoder_3 = self.encoder_3(self.encoder_maxpool_2(encoder_2))
        encoder_4 = self.encoder_4(self.encoder_maxpool_3(encoder_3))
        encoder_5 = self.encoder_5(self.encoder_maxpool_4(encoder_4))
        encoder_5 = self.calculate_pad(
            encoder_4, self.decoder_upsampling_1(encoder_5))
        decoder_1 = self.decoder_1(torch.cat([encoder_4, encoder_5], dim=1))
        decoder_1 = self.calculate_pad(
            encoder_3, self.decoder_upsampling_2(decoder_1))
        decoder_2 = self.decoder_2(torch.cat([encoder_3, decoder_1], dim=1))
        decoder_2 = self.calculate_pad(
            encoder_2, self.decoder_upsampling_3(decoder_2))
        decoder_3 = self.decoder_3(torch.cat([encoder_2, decoder_2], dim=1))
        decoder_3 = self.calculate_pad(
            encoder_1, self.decoder_upsampling_4(decoder_3))
        decoder_4 = self.decoder_4(torch.cat([encoder_1, decoder_3], dim=1))
        output = self.output(decoder_4)
        return output

if __name__ == '__main__':
    pass
