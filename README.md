# unet-to-segment-camvid

## Overview

This project is a simple implementation of a UNet for semantic segmentation of the CamVid dataset.

## Dataset

The dataset can be found [here](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid).
Images and masks are originally 960x720 in size.

### Data Preprocessing

The images and masks are processed to be 720x720 in size.
Each mask/image is cropped to be square, in this case we have an overlap of 480 pixels (50% of the image size).






