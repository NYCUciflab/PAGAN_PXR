import torch
import torch.nn as nn
from .module import calculate_channels, decode_layer, upsampling_block


class Discriminator(nn.Module):
    def __init__(self, model, input_shape=64, sigmoid=False):
        super().__init__()

        # define layer information
        input_channels = calculate_channels(model, inupt_shape=input_shape)
        input_shapes = [input_shape//2, input_shape//4, input_shape//8, input_shape//16]
        self.input_shape = input_shape
        self.sigmoid = sigmoid

        self.model        = model
        self.eff_layer_1  = nn.Sequential(*list(model.features.children())[0:2])
        self.eff_layer_2  = nn.Sequential(*list(model.features.children())[2:3])
        self.eff_layer_3  = nn.Sequential(*list(model.features.children())[3:4])
        self.eff_layer_4  = nn.Sequential(*list(model.features.children())[4:5])
        self.eff_layer_5  = nn.Sequential(*list(model.features.children())[5:])
        self.avgpool      = nn.AdaptiveAvgPool2d((1, 1))

        # upsampling block
        self.up_4_from_5   = upsampling_block(input_channels[-1], 256)
        self.up_3_from_4   = upsampling_block(256, 128)
        self.up_2_from_3   = upsampling_block(128, 64)
        self.up_1_from_2   = upsampling_block(64, 32)
        self.output_from_1 = nn.Sequential(upsampling_block(32, 3), nn.Tanh())

        # D skip output
        self.dropout = nn.Dropout2d(p=0.25)
        self.D_skip = nn.Conv2d(input_channels[3], 3, kernel_size=3, stride=1, padding='same', bias=True)
        self.D_out  = nn.Conv2d(3, 1, kernel_size=input_shapes[3], stride=1, padding=0, bias=True)

        # C skip output 
        self.C_skip = nn.Conv2d(input_channels[3], 3, kernel_size=3, stride=1, padding='same', bias=True)
        self.C_out  = nn.Conv2d(3, 1, kernel_size=input_shapes[3], stride=1, padding=0, bias=True)


    def forward(self, x):
        x1 = self.eff_layer_1(x)
        x2 = self.eff_layer_2(x1)
        x3 = self.eff_layer_3(x2)
        x4 = self.eff_layer_4(x3)
        x5 = self.eff_layer_5(x4)


        # upsampling block
        up_4 = self.up_4_from_5(x5)
        up_3 = self.up_3_from_4(up_4)
        up_2 = self.up_2_from_3(up_3)
        up_1 = self.up_1_from_2(up_2)
        mse_out = self.output_from_1(up_1)

        # skip output
        
        D_skip = self.D_skip(x4)
        D_out  = self.D_out(D_skip)
        D_out  = D_out.view(D_out.shape[0], -1)
        if self.sigmoid:
            D_out = torch.sigmoid(D_out)

        x4     = self.dropout(x4)
        C_skip = self.C_skip(x4)
        C_out  = self.C_out(C_skip)
        C_out  = C_out.view(C_out.shape[0], -1)
    
        return D_out, C_out, mse_out
    




# Patch Auxiliary Generator
class PAGenerator(nn.Module):
    def __init__(self, model, input_shape=512):
        super().__init__()

        # define layer information
        input_channels = calculate_channels(model, inupt_shape=input_shape)
        input_shapes = [input_shape, input_shape//2, input_shape//4, input_shape//8, input_shape//16, input_shape//32]
        self.input_shape = input_shape

        self.model        = model
        self.eff_layer_1  = nn.Sequential(*list(model.features.children())[0:2])
        self.eff_layer_2  = nn.Sequential(*list(model.features.children())[2:3])
        self.eff_layer_3  = nn.Sequential(*list(model.features.children())[3:4])
        self.eff_layer_4  = nn.Sequential(*list(model.features.children())[4:5])
        self.eff_layer_5  = nn.Sequential(*list(model.features.children())[5:6])
        self.eff_layer_6  = nn.Sequential(*list(model.features.children())[6:7])
        self.eff_layer_7  = nn.Sequential(*list(model.features.children())[7:])
        # dropout
        self.dropout_6      = nn.Dropout(0.5)
        self.dropout_7      = nn.Dropout(0.5)
        self.dropout_8      = nn.Dropout(0.5)

        self.avgpool      = nn.AdaptiveAvgPool2d((1, 1))
        self.class_out    = nn.Linear(input_channels[-1], 1)

        # skip output
        channel_idx = -1

        # skip 6
        self.depthwise_1 = nn.Conv2d(
            input_channels[channel_idx-1], 
            input_channels[channel_idx-1], 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            groups=input_channels[channel_idx-1], 
            bias=True
            )

        self.pointwise_1 = nn.Conv2d(
            input_channels[channel_idx-1], 
            1024, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=True
            )
        

        # decoder
        decode_dim = [input_shapes[0]*2, input_shapes[0], input_shapes[1], input_shapes[2], input_shapes[3]]
        init_kernelSize = input_shapes[-1]
        decode_kernelSize = [init_kernelSize+1, 4, 4, 4, 4]
        
        self.decode_layer_1 = decode_layer(decode_dim[0], decode_dim[1], kernel_size=decode_kernelSize[0], stride=1, padding=0, output_padding=0)
        self.decode_layer_2 = decode_layer(decode_dim[1], decode_dim[2], kernel_size=decode_kernelSize[1]+1, stride=1, padding=2, output_padding=0)
        self.decode_layer_3 = decode_layer(decode_dim[2], decode_dim[3], kernel_size=decode_kernelSize[2], stride=2, padding=1, output_padding=0)
        self.decode_conv    = nn.ConvTranspose2d(decode_dim[3], 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.decode_actv    = nn.Tanh()


    def forward(self, x):
        x1 = self.eff_layer_1(x)
        x2 = self.eff_layer_2(x1)
        x3 = self.eff_layer_3(x2)
        x4 = self.eff_layer_4(x3)
        x5 = self.eff_layer_5(x4) #x5 32x32
        x6 = self.eff_layer_6(x5) #x6 16x16
        x6 = self.dropout_6(x6) 
        x7 = self.eff_layer_7(x6) #x7 16x16
        x7 = self.dropout_7(x7)
        x8 = self.avgpool(x7)
        x8 = self.dropout_8(x8)
        out = self.class_out(x8.view(x8.shape[0], -1))

        # skip output
        skip = self.depthwise_1(x6)
        skip = self.pointwise_1(skip)

        # decoder
        gen = self.decode_layer_1(skip)
        gen = self.decode_layer_2(gen)
        gen = self.decode_layer_3(gen)
        gen = self.decode_conv(gen)
        gen = self.decode_actv(gen)
    
        return out, gen
    


# efficientnet-b0
class class_model(nn.Module):
    def __init__(self, model, input_shape=512):
        super().__init__()
        # define layer information
        input_channels = calculate_channels(model, inupt_shape=input_shape)
        self.input_shape = input_shape

        self.model        = model
        self.eff_layer_1  = nn.Sequential(*list(model.features.children())[0:2])
        self.eff_layer_2  = nn.Sequential(*list(model.features.children())[2:3])
        self.eff_layer_3  = nn.Sequential(*list(model.features.children())[3:4])
        self.eff_layer_4  = nn.Sequential(*list(model.features.children())[4:5])
        self.eff_layer_5  = nn.Sequential(*list(model.features.children())[5:6])
        self.eff_layer_6  = nn.Sequential(*list(model.features.children())[6:7])
        self.eff_layer_7  = nn.Sequential(*list(model.features.children())[7:])
        self.avgpool      = nn.AdaptiveAvgPool2d((1, 1))
        self.class_out    = nn.Linear(input_channels[-1], 1)


    def forward(self, x):
        x1 = self.eff_layer_1(x)
        x2 = self.eff_layer_2(x1)
        x3 = self.eff_layer_3(x2)
        x4 = self.eff_layer_4(x3)
        x5 = self.eff_layer_5(x4) #x5 32x32
        x6 = self.eff_layer_6(x5) #x6 16x16
        x7 = self.eff_layer_7(x6) #x7 16x16
        x8 = self.avgpool(x7)
        out = self.class_out(x8.view(x8.shape[0], -1))

        return out