import mypix2pix
if __name__ == '__main__':
    model=mypix2pix.pix2pix(0.0002,0.5,"C:\\Users\\long\\Desktop\\pytorch-CycleGAN-and-pix2pix\\model_save","C:\\Users\\long\\Desktop\\myd-master\\myd-master","C:\\Users\\long\\Desktop\\pytorch-CycleGAN-and-pix2pix\\rz")
    model.start_train(10000)