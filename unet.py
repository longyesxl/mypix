import torch
import torch.nn as nn
import torch.nn.functional as F
# class Unet(nn.Module):
#     def __init__(self, input_nc, output_nc, num_downs, ngf=64,maxngf=512, norm_layer=nn.BatchNorm2d, use_dropout=False,start_conv=True):
#         super(Unet, self).__init__()
#         self.ngf=min(ngf,maxngf)
#         self.ngfm=min(ngf*2,maxngf)
#         if(num_downs>1):
#             self.submodel=Unet(self.ngf,self.ngf,num_downs-1,ngf=self.ngfm,maxngf=maxngf,norm_layer=norm_layer,use_dropout=use_dropout,start_conv=False)
#         use_bias = norm_layer == nn.InstanceNorm2d
#         self.start_conv=start_conv
#         self.num_downs=num_downs
#         self.use_dropout=use_dropout
#         self.input_nc=input_nc
#         self.output_nc=output_nc
#         self.downconv = nn.Conv2d(input_nc, self.ngf, kernel_size=4,
#                              stride=2, padding=1, bias=use_bias)
#         self.downrelu = nn.LeakyReLU(0.2, True)
#         self.downnorm = norm_layer(self.ngf)
#         self.uprelu = nn.ReLU(True)
#         self.upnorm = norm_layer(output_nc)
#         if self.num_downs>1:
#             self.upconv = nn.ConvTranspose2d(self.ngf*2, output_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1)
#         else:
#             self.upconv = nn.ConvTranspose2d(self.ngf, output_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1)
#         if use_dropout:
#             self.updropout=nn.Dropout(0.5)
#     def forward(self, x):
#         if self.start_conv:
#             upthanh=nn.Tanh()
#             y=self.downconv(x)
#             y=self.submodel(y)
#             y=self.uprelu(y)
#             y=self.upconv(y)
#             y=upthanh(y)
#             if self.use_dropout:
#                 y=self.updropout(y)
#             return y 
            
#         elif self.num_downs>1:
#             y=self.downrelu(x)
#             y=self.downconv(y)
#             y=self.downnorm(y)
#             y=self.submodel(y)
#             y=self.uprelu(y)
#             y=self.upconv(y)
#             y=self.upnorm(y)
#             if self.use_dropout:
#                 y=self.updropout(y)
#             return torch.cat([x, y], 1) 
#         else:   # add skip connections
#             y=self.downrelu(x)
#             y=self.downconv(y)
#             y=self.uprelu(y)
#             y=self.upconv(y)
#             y=self.upnorm(y)
#             if self.use_dropout:
#                 y=self.updropout(y)
#             return torch.cat([x, y], 1) 

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)   