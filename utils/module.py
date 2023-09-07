import torch
import torch.nn as nn




def calculate_channels(model, inupt_shape=64):
    channels = []
    test_noise = torch.randn(1, 3, inupt_shape, inupt_shape)
    x = nn.Sequential(*list(model.features.children())[0:2])(test_noise)
    channels.append(x.shape[1])
    x = nn.Sequential(*list(model.features.children())[2:3])(x)
    channels.append(x.shape[1])
    x = nn.Sequential(*list(model.features.children())[3:4])(x)
    channels.append(x.shape[1])
    x = nn.Sequential(*list(model.features.children())[4:5])(x)
    channels.append(x.shape[1])
    x = nn.Sequential(*list(model.features.children())[5:6])(x)
    channels.append(x.shape[1])
    x = nn.Sequential(*list(model.features.children())[6:7])(x)
    channels.append(x.shape[1])
    x = nn.Sequential(*list(model.features.children())[7:])(x)
    channels.append(x.shape[1])

    return channels




def decode_layer(input_dim, output_dim, kernel_size=4, stride=1, padding=1, output_padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, output_padding),
        nn.BatchNorm2d(output_dim),
        # nn.ReLU()
    )




def upsampling_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.BatchNorm2d(out_channels)
    )