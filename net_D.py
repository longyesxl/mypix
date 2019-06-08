import torch
import torch.nn as nn
import torch.nn.functional as F

class net_D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(net_D, self).__init__()
        use_bias = norm_layer != nn.BatchNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        self.model1 = nn.Sequential(*sequence).cuda()
        self.modellist=[]
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence= [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            model = nn.Sequential(*sequence).cuda()
            self.modellist.append(model)

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence = [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        self.model2 = nn.Sequential(*sequence).cuda()
        sequence = [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model3 = nn.Sequential(*sequence).cuda()

    def forward(self, input):
        """Standard forward."""
        output=[]
        out=self.model1(input)
        output.append(out)
        for hh in self.modellist:
            out=hh(out)
            output.append(out)
        out=self.model2(out)
        output.append(out)
        out=self.model3(out)
        output.append(out)
        
        return output

