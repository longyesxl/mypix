import net_D,unet
import torch
import torch.nn as nn
import os
import mydataset
from torch.utils.data import DataLoader
import random
import numpy as np
import cv2
import tqdm
from torch.nn import init

class pix2pix():
    def __init__(self,lr,beta1,model_path,data_path,result_path):
        self.device = torch.device('cuda:0')
        self.train_dataset=mydataset.myDataset(data_path)
        # self.test_dataset=mydataset.myDataset(data_path)
        self.val_dataset=mydataset.myDataset(data_path,val_set=True)
        self.result_path=result_path
        self.model_path=model_path
        self.data_path=data_path
        self.net_G=GlobalGenerator(3,3).to(self.device)
        #self.net_G=unet.Unet(3,3,8).to(self.device)
        self.net_D=net_D.net_D(6).to(self.device)
        self.init_weights()
        if os.path.exists(model_path+"/net_G.pth"):
            self.net_G.load_state_dict(torch.load(model_path+"/net_G.pth"))
        if os.path.exists(model_path+"/net_D.pth"):
            self.net_D.load_state_dict(torch.load(model_path+"/net_D.pth"))
        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=lr, betas=(beta1, 0.999))
        self.gan_loss=nn.MSELoss().to(self.device)
        self.l1_loss = nn.L1Loss().to(self.device)
        self.real_label=torch.tensor(1.0)
        self.fake_label=torch.tensor(0.0)
    def init_weights(self, init_type='normal', init_gain=0.02):
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)
        print('initialize network with %s' % init_type)
        self.net_G.apply(init_func)
        self.net_D.apply(init_func)
    def forward(self,input):
        self.real_in = input['img_in'].to(self.device)
        self.real_out = input['img_out'].to(self.device)
        self.fake_out = self.net_G(self.real_in)
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake = torch.cat((self.real_in, self.fake_out), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.net_D(fake.detach())
        pred_fake=pred_fake[len(pred_fake)-1]
        self.loss_D_fake = self.gan_loss(pred_fake, self.fake_label.expand_as(pred_fake).to(self.device))
        # Real
        real = torch.cat((self.real_in,self.real_out), 1)
        pred_real = self.net_D(real)
        pred_real=pred_real[len(pred_real)-1]
        self.loss_D_real = self.gan_loss(pred_real, self.real_label.expand_as(pred_real).to(self.device))
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake = torch.cat((self.real_in, self.fake_out), 1)
        real = torch.cat((self.real_in,self.real_out), 1)
        pred_fake = self.net_D(fake)
        pred_real = self.net_D(real)
        # Second, G(A) = B
        self.loss_G_L1 = self.l1_loss(self.fake_out, self.real_out) 
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_L1
        for i in range(len(pred_fake)):
            self.loss_G+=self.gan_loss(pred_fake[i], pred_real[i])/len(pred_fake)
        self.loss_G.backward()

    def train(self,input):
        self.forward(input)                   # compute fake images: G(A)
        # update D
        self.net_D.requires_grad=True # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.net_D.requires_grad=False  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
    def test(self,input):
        self.forward(input)                   # compute fake images: G(A)
        # update D
        self.net_D.requires_grad=False # enable backprop for D
        self.backward_D()                # calculate gradients for D
        self.backward_G()                   # calculate graidents for G
    
    def save_model(self):
        torch.save(self.net_G.state_dict(),self.model_path+"/net_G.pth")
        torch.save(self.net_D.state_dict(),self.model_path+"/net_D.pth")
        
    def start_train(self,epoch_nub):
        for epoch in tqdm.tqdm(range(epoch_nub)):
            trian_dataloader = DataLoader(self.train_dataset, batch_size=4,shuffle=True, num_workers=4)
            train_G_loss=0.0
            train_D_loss=0.0
            test_G_loss=0.0
            test_D_loss=0.0
            for i_batch, sample_batched in enumerate(trian_dataloader):
                self.train(sample_batched)
                train_G_loss+=self.loss_G.item()
                train_D_loss+=self.loss_D.item()
            print("train_G_loss:"+str(train_G_loss/(i_batch+1))+"\t"+"train_D_loss:"+str(train_D_loss/(i_batch+1)))
            r_in=self.real_in.cpu().numpy().transpose((0,2,3,1))[0]
            r_out=self.real_out.cpu().numpy().transpose((0,2,3,1))[0]
            f_out=self.fake_out.detach().cpu().clamp(0.0,1.0).numpy().transpose((0,2,3,1))[0]
            result=np.concatenate((r_in, r_out,f_out),axis=1)*255
            rz=result.astype(np.uint8)
            cv2.imwrite(self.result_path+"/rz_img/train/"+("%05d" % epoch)+".jpg",rz)
            self.save_model()

            val_dataloader = DataLoader(self.val_dataset, batch_size=1,shuffle=False, num_workers=1)
            for i_batch, sample_batched in enumerate(val_dataloader):
                self.test(sample_batched)
                test_G_loss+=self.loss_G.item()
                test_D_loss+=self.loss_D.item()
            print("val_G_loss:"+str(test_G_loss/(i_batch+1))+"\t"+"val_D_loss:"+str(test_D_loss/(i_batch+1)))
            r_in=self.real_in.cpu().numpy().transpose((0,2,3,1))[0]
            r_out=self.real_out.cpu().numpy().transpose((0,2,3,1))[0]
            f_out=self.fake_out.detach().cpu().clamp(0.0,1.0).numpy().transpose((0,2,3,1))[0]
            result=np.concatenate((r_in, r_out,f_out),axis=1)*255
            rz=result.astype(np.uint8)
            cv2.imwrite(self.result_path+"/rz_img/val/"+("%05d" % epoch)+".jpg",rz) 
