import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pandas as pd
from torch.optim import Adam
from .loss import *


class train_by_epoch:
    
    def __init__(self, 
                 testing_code,
                 generator, 
                 discriminator, 
                 epochs, 
                 batchs,
                 train_dataset,
                 valid_dataset,
                 crop_dataset, 
                 device,
                 result_dir,
                 weight_dir,
                 classifier_loss,
                 gan_learning_rate=0.00005,
                 classifier_learning_rate=0.002,
                 weight_clipping=False, 
                 GAN_loss='lsgan',
                 gradient_panelty=10, 
                 original_gan=False,
                 ):
        
        self.testing_code = testing_code
        self.G       = generator
        self.D       = discriminator
        self.epochs  = epochs
        self.batchs  = batchs
        self.device  = device

        self.crop_dataset = crop_dataset
        self.gan_learning_rate = gan_learning_rate
        self.b1 = 0.0
        self.b2 = 0.9
        self.G_optim = Adam(self.G.parameters(), lr=self.gan_learning_rate, betas=(self.b1, self.b2))
        self.D_optim = Adam(self.D.parameters(), lr=self.gan_learning_rate, betas=(self.b1, self.b2))
        self.weight_clipping_limit = weight_clipping
        self.gradient_panelty = gradient_panelty
        self.result_dir = result_dir
        self.weight_dir = weight_dir
        self.original_gan = original_gan

        # define gan loss
        self.GAN_loss = GAN_loss
        if self.GAN_loss == 'wgan':
            self.G_loss = WGAN_G_loss
            self.D_loss = WGAN_D_loss

        elif self.GAN_loss == 'lsgan':
            self.G_loss = LSGAN_G_loss
            self.D_loss = LSGAN_D_loss

        elif self.GAN_loss == 'gan':
            self.G_loss = GAN_G_loss
            self.D_loss = GAN_D_loss


        # for classifier
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.classifier_learning_rate = classifier_learning_rate
        self.classifier_loss = classifier_loss
        self.train_step = len(self.train_dataset)
        self.valid_step = len(self.valid_dataset)

        # define a fix image for visualization
        fix_img = next(iter(self.train_dataset))['image']
        self.fix_img    = fix_img.to(self.device)
        self.img_shape  = fix_img.shape

        if self.testing_code:
            self.train_step = 10
            self.valid_step = 5


        
    def update_G(self, fullsize_img, label):        
        for p in self.D.parameters():
            p.requires_grad = False
            
        for p in self.G.parameters():
            p.requires_grad = True
        
        self.G_optim.zero_grad()        

        # input image to G to generate fake patch image
        c_out, fake_img = self.G(fullsize_img)
        fake_img = fake_img.repeat(1,3,1,1)

        # input fake patch to D
        D_fake_output, D_label, _ = self.D(fake_img)

        # calculate loss
        loss_G = self.G_loss(D_fake_output.reshape(-1), original=self.original_gan)
        loss_C = self.classifier_loss(c_out, label)
        lossDC = self.classifier_loss(D_label, label)
        loss   = loss_G + 0.5*loss_C + 0.5*lossDC
        
        # update G
        loss.backward()
        self.G_optim.step()
        
        return loss_G, loss_C
        
        
        
    def update_D(self, fullsize_img, fullsize_label, crop_img):        
        for p in self.D.parameters():
            p.requires_grad = True
            
        for p in self.G.parameters():
            p.requires_grad = False
        
        self.D_optim.zero_grad()

        if self.weight_clipping_limit:
            for p in self.D.parameters():
                p.data.clamp_(-self.weight_clipping_limit, self.weight_clipping_limit)

        # input image to G to generate fake patch image
        _, fake_img         = self.G(fullsize_img)
        fake_img            = fake_img.repeat(1,3,1,1)

        # input fake patch and real image to D
        D_real, _, D_decode = self.D(crop_img)
        D_fake, D_label, _  = self.D(fake_img)
        
        # calculate loss
        loss_D   = self.D_loss(D_real, D_fake, original=self.original_gan)
        loss_C   = self.classifier_loss(D_label, fullsize_label)
        mse_loss = nn.MSELoss()(D_decode, crop_img) + nn.L1Loss()(D_decode, crop_img)
        loss     = loss_D + loss_C + mse_loss

        # gradient panelty
        if self.gradient_panelty:
            gp = self._gradient_panelty(crop_img, fake_img)
            loss = loss + self.gradient_panelty*gp
            
        # update D
        loss.backward(retain_graph=True)
        self.D_optim.step()
        
        return loss_D
    

    def _gradient_panelty(self, real_img, fake_img):
        
        batch, c, h, w = real_img.shape
        epsilon = torch.rand((batch, 1, 1, 1)).repeat(1,c,h,w).to(self.device)
        interpolated_imgs = (real_img*epsilon + fake_img*(1-epsilon)).requires_grad_(True)
        
        D_interpolated, _, _ = self.D(interpolated_imgs)
        gradient = torch.autograd.grad(
            outputs=D_interpolated, 
            inputs=interpolated_imgs, 
            grad_outputs=torch.ones_like(D_interpolated), 
            create_graph=True, 
            retain_graph=True)[0]
        
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_panelty = torch.mean((gradient_norm-1)**2)
        
        return gradient_panelty
       
        
        
        
    def train(self):
        dlosses = []
        glosses = []
        closses = []
        global_best_ydn = 0

        #  losses
        avg_valid_losses = []
        avg_test_losses  = []

        accs = []
        sens = []
        spcs = []
        ydns = []

        print('start training...')

        for epoch in range(self.epochs):
            start = time.time()
            d_batch_loss = 0
            g_batch_loss = 0
            c_batch_loss = 0
            
            # update D
            self.D.train()
            self.G.train()
            for batch_idx, data in enumerate(self.train_dataset):
                full_size_imgs = data['image'].to(self.device)
                labels         = data['label'].to(self.device)
                crop_img       = next(iter(self.crop_dataset)).to(self.device)
                
                # update D
                d_loss = self.update_D(full_size_imgs, labels, crop_img)
                d_loss = d_loss.cpu().detach().numpy()
                dloss = round(d_loss.item(), 4)
                d_batch_loss += dloss

                # update G                
                g_loss, c_loss = self.update_G(full_size_imgs, labels)
                g_loss = g_loss.cpu().detach().numpy()
                c_loss = c_loss.cpu().detach().numpy()
                gloss = round(g_loss.item(), 4)
                closs = round(c_loss.item(), 4)
                g_batch_loss += gloss
                c_batch_loss += closs

                

                print(f'EPOCH: [{epoch+1}/{self.epochs}]  BATCH: [{batch_idx+1}/{len(self.train_dataset)}]  D_loss:{dloss}   G_loss: {gloss}   C_loss: {closs}', end='\r')

                if self.testing_code:
                    if batch_idx==4:
                        break
            
            mean_dloss = round(d_batch_loss/(batch_idx+1), 4)
            mean_gloss = round(g_batch_loss/(batch_idx+1), 4)
            mean_closs = round(c_batch_loss/(batch_idx+1), 4)
            time_cost = round(time.time()-start, 1)
            print(f'EPOCH: [{epoch+1}/{self.epochs}]   D_loss:{mean_dloss}    G_loss: {mean_gloss}      C_loss: {mean_closs}     Time:{time_cost}s   ')

            dlosses.append(dloss)
            glosses.append(gloss)
            closses.append(closs)


            # validation
            self.G.eval()
            self.D.eval()
            valid_losses = []
            labels       = []
            preds        = []
            for i_batch, sample_batched in enumerate(self.valid_dataset):

                imgs   = sample_batched['image'].to(self.device)
                labels = sample_batched['label'].to(self.device)

                self.G_optim.zero_grad()
                pred, _ = self.G(imgs)
                loss = self.classifier_loss(pred, labels)

                # append losses
                valid_loss = loss.tolist()
                valid_losses.append(valid_loss)

                # calculate average losses
                avg_valid_loss = np.mean(valid_losses)

                labels.append(label.tolist()[0])
                preds.append(pred.tolist()[0][0])
                
                if self.testing_code:
                    if i_batch+1 == 5:
                        break
                        

                
            avg_valid_loss = np.mean(valid_losses)
            avg_valid_losses.append(avg_valid_loss)

            acc, sen, spc, ydn = get_results(preds, labels, threshold=0.5)
            print(f'acc: %.4f   sen: %.4f   spc:%.4f   ydn:%.4f'%(acc, sen, spc, ydn))
            accs.append(acc)
            sens.append(sen)
            spcs.append(spc)
            ydns.append(ydn)
            
            saved_info = pd.DataFrame({'train loss': closses,
                                    'valid loss': avg_valid_losses,
                                    'dloss':dlosses,
                                    'gloss':glosses,
                                    'acc': accs,
                                    'sen': sens,
                                    'spc': spcs,
                                    'ydn': ydns
                                    })
            
            
            saved_info.to_csv(self.result_dir+'results.csv', index=0)

            
            Saved_ydn  = round(ydn, 4)
            Saved_loss = round(avg_valid_loss, 4)


            if Saved_ydn > global_best_ydn:
                torch.save(self.G.state_dict(), self.weight_dir+f'G/epoch_{epoch+1}_loss_{Saved_loss}_ydn_{Saved_ydn}.pt')
                torch.save(self.D.state_dict(), self.weight_dir+f'D/D_epoch_{epoch+1}.pt')
                global_best_ydn = Saved_ydn
           
            generated_img = self.get_generate_image(row_number=4)[...,0]
            generated_img = Image.fromarray((generated_img*255).astype(np.uint8))
            generated_img.save(self.result_dir+'saved_img/'+f"G_img_epoch_{epoch+1}.png")
            
                        
                    
        
    def get_generate_image(self, row_number=4):

        _, g_img = self.G(self.fix_img)
        g_img = g_img.cpu().detach()
        g_arr = g_img.permute((0,2,3,1)).numpy()
        g_arr = (g_arr+1)/2
        
        reshape_arr = np.hstack(g_arr)
        img_size    = reshape_arr.shape[0]
        col_number  = reshape_arr.shape[1]//reshape_arr.shape[0]//row_number
        reshape_arr_zero = np.zeros((img_size*row_number, img_size*col_number, 3))
        
        for i in range(row_number):
            reshape_arr_zero[i*img_size:(i+1)*img_size,...] = \
            reshape_arr[:,(i*img_size*col_number):((i+1)*img_size*col_number),:]
            
        return reshape_arr_zero
