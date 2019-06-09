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
from torch.autograd import Variable

class pix2pix():
    def __init__(self,lr,beta1,model_path,data_path,result_path,net_G_type="unet"):
        self.device = torch.device('cuda:0')
        self.train_dataset=mydataset.myDataset(data_path)
        # self.test_dataset=mydataset.myDataset(data_path)
        self.val_dataset=mydataset.myDataset(data_path,val_set=True)
        self.result_path=result_path
        self.model_path=model_path
        self.data_path=data_path
        if(net_G_type=="unet"):
            self.net_G=unet.Unet(3,3,8).to(self.device)
        else:
            self.net_G=unet.GlobalGenerator(3,3,8).to(self.device)
        #self.net_D=net_D.net_D(6).to(self.device)
        self.net_D=net_D.MultiscaleDiscriminator(6, 64, 3, nn.BatchNorm2d, False, 1, True).to(self.device)
        self.init_weights()
        if os.path.exists(model_path+"/net_G.pth"):
            self.net_G.load_state_dict(torch.load(model_path+"/net_G.pth"))
        if os.path.exists(model_path+"/net_D.pth"):
            self.net_D.load_state_dict(torch.load(model_path+"/net_D.pth"))
        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=lr, betas=(beta1, 0.999))
        self.Tensor = torch.cuda.FloatTensor
        self.criterionGAN = GANLoss(use_lsgan=True,tensor=self.Tensor).to(self.device)
        self.criterionFeat = torch.nn.L1Loss().to(self.device)
        self.real_label=torch.tensor(1.0)
        self.fake_label=torch.tensor(0.0)
        self.fake_pool = ImagePool(0)
        self.criterionVGG = VGGLoss(0).to(self.device)
    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.net_D.forward(fake_query)
        else:
            return self.net_D.forward(input_concat)
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
        pred_fake = self.discriminate(self.real_in, self.fake_out)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)   
        # Real
        pred_real = self.discriminate(self.real_in, self.real_out)
        self.loss_D_real = self.criterionGAN(pred_real, True)   
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        pred_fake = self.net_D.forward(torch.cat((self.real_in, self.fake_out), dim=1))        
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) 
        pred_real = self.discriminate(self.real_in, self.real_out)

        self.loss_G_GAN_Feat = 0
        feat_weights = 1.0
        D_weights = 1.0
        for i in range(1):
            for j in range(len(pred_fake[i])-1):
                self.loss_G_GAN_Feat += D_weights * feat_weights * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * 10.0
        # Second, G(A) = B
        self.loss_G_VGG = self.criterionVGG(self.fake_out, self.real_out) * 10.0
        self.loss_G=self.loss_G_GAN+self.loss_G_GAN_Feat+self.loss_G_VGG
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
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)
from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()       
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss
class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images
