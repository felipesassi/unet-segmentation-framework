
import sys
import os.path

path = os.path.dirname(os.getcwd())
sys.path.append(path)

from models.unet_2d import UNet2D
import torch


def test_output_shape():
    batch_size = 2
    in_channels = 10
    out_channels = 10
    filter_size = 10
    img_size = (240, 240)
    UNet = UNet2D(in_channels, out_channels, filter_size)
    assert list(UNet(torch.zeros([batch_size, in_channels, img_size[0], img_size[1]])).shape) == [
        batch_size, in_channels, img_size[0], img_size[1]]


if __name__ == "__main__":
    print("Test file.")
