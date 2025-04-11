'''import torch
from torch.nn import functional as F
from conformer import build_model
import numpy as np
import os
import cv2
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
writer = SummaryWriter('log/run' + time.strftime("%d-%m"))
import torch.nn as nn
import argparse
import os.path as osp
import os
size_coarse = (10, 10)
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from  utils import  count_model_flops,count_model_params
from PIL import Image
import json



class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        #self.build_model()
        self.net = build_model(self.config.network, self.config.arch)
        #self.net.eval()
        if config.mode == 'test':
            print('Loading pre-trained model for testing from %s...' % self.config.model)
            self.net.load_state_dict(torch.load(self.config.model, map_location=torch.device('cpu')))
        if config.mode == 'train':
            if self.config.load == '':
                print("Loading pre-trained imagenet weights for fine tuning")
                self.net.JLModule.load_pretrained_model(self.config.pretrained_model
                                                        if isinstance(self.config.pretrained_model, str)
                                                        else self.config.pretrained_model[self.config.network])
                # load pretrained backbone
            else:
                print('Loading pretrained model to resume training')
                self.net.load_state_dict(torch.load(self.config.load))  # load pretrained model
        
        if self.config.cuda:
            self.net = self.net.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net, 'Conformer based SOD Structure')

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params_t = 0
        num_params=0
        for p in model.parameters():
            if p.requires_grad:
                num_params_t += p.numel()
            else:
                num_params += p.numel()
        print(name)
        #print(model)
        print("The number of trainable parameters: {}".format(num_params_t))
        print("The number of parameters: {}".format(num_params))
        print(f'Flops: {count_model_flops(model)}')
        print(f'Flops: {count_model_params(model)}')

    # build the network
    def build_model(self):
        self.net = build_model(self.config.network, self.config.arch)

        if self.config.cuda:
            self.net = self.net.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net, 'JL-DCF Structure')

    def test(self):
        print('Testing...')
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size, depth = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size']), \
                                           data_batch['depth']
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device)
                    depth = depth.to(device)

                #input = torch.cat((images, depth), dim=0)
                preds,coarse_sal_rgb,e_rgbd0,lde_out,rgb_h,rgb_m= self.net(images)
                #print(depth_1.shape[1])
                #rgb_1_a=((torch.sum(rgb_1,1)**2)/rgb_1.shape[1]).unsqueeze(0)
                #print(depth_1.shape)
                rgb_2_a=(torch.sum(rgb_2,1)/rgb_2.shape[1]).unsqueeze(0)
                rgb_3_a=(torch.sum(rgb_3,1)/rgb_3.shape[1]).unsqueeze(0)
                rgb_4_a=(torch.sum(rgb_4,1)/rgb_4.shape[1]).unsqueeze(0)
                rgb_5_a=(torch.sum(rgb_5,1)/rgb_5.shape[1]).unsqueeze(0)
                depth_1_a=((torch.sum(depth_1,1)**2)/depth_1.shape[1]).unsqueeze(0)
                depth_2_a=((torch.sum(depth_2,1)**2)/depth_2.shape[1]).unsqueeze(0)
                depth_3_a=(torch.sum(depth_3,1)/depth_3.shape[1]).unsqueeze(0)
                depth_4_a=(torch.sum(depth_4,1)/depth_4.shape[1]).unsqueeze(0)
                depth_5_a=(torch.sum(depth_5,1)/depth_5.shape[1]).unsqueeze(0)
                rgbd_fusion_1_a=(torch.sum(rgbd_fusion_1,1)/rgbd_fusion_1.shape[1]).unsqueeze(0)
                rgbd_fusion_2_a=(torch.sum(rgbd_fusion_2,1)/rgbd_fusion_2.shape[1]).unsqueeze(0)
                rgbd_fusion_3_a=(torch.sum(rgbd_fusion_3,1)/rgbd_fusion_3.shape[1]).unsqueeze(0)
                rgbd_fusion_4_a=(torch.sum(rgbd_fusion_4,1)/rgbd_fusion_4.shape[1]).unsqueeze(0)
                rgbd_fusion_5_a=(torch.sum(rgbd_fusion_5,1)/rgbd_fusion_5.shape[1]).unsqueeze(0)
                rgb_1_a = F.interpolate(rgb_1_a, tuple(im_size), mode='bilinear', align_corners=True)
                rgb_1_a = np.squeeze(torch.sigmoid(rgb_h)).cpu().data.numpy()
                rgb_1_a = (rgb_1_a- rgb_1_a.min()) / (rgb_1_a.max() - rgb_1_a.min() + 1e-8)
                multi_fuse_rgb_1_a = 255 * rgb_1_a
                #multi_fuse_rgb_1_a=cv2.applyColorMap((multi_fuse_rgb_1_a).astype(np.uint8), cv2.COLORMAP_RAINBOW)
                filename_rgb_1_a = os.path.join(self.config.test_folder, name[:-4] + '_rgb_1_a.png')
                cv2.imwrite(filename_rgb_1_a, multi_fuse_rgb_1_a)

                #rgb_2_a = F.interpolate(rgb_2_a, tuple(im_size), mode='bilinear', align_corners=True)
                rgb_2_a = np.squeeze(torch.sigmoid(rgb_m)).cpu().data.numpy()
                rgb_2_a = (rgb_2_a- rgb_2_a.min()) / (rgb_2_a.max() - rgb_2_a.min() + 1e-8)
                multi_fuse_rgb_2_a = 255 * rgb_2_a
                #multi_fuse_rgb_2_a=cv2.applyColorMap((multi_fuse_rgb_2_a).astype(np.uint8), cv2.COLORMAP_RAINBOW)
                filename_rgb_2_a = os.path.join(self.config.test_folder, name[:-4] + '_rgb_2_a.png')
                cv2.imwrite(filename_rgb_2_a, multi_fuse_rgb_2_a)

                #rgb_3_a = F.interpolate(rgb_3_a, tuple(im_size), mode='bilinear', align_corners=True)
                rgb_3_a = np.squeeze(torch.sigmoid(lde_out)).cpu().data.numpy()
                rgb_3_a = (rgb_3_a- rgb_3_a.min()) / (rgb_3_a.max() - rgb_3_a.min() + 1e-8)
                multi_fuse_rgb_3_a = 255 * rgb_3_a
                #multi_fuse_rgb_3_a=cv2.applyColorMap((multi_fuse_rgb_3_a).astype(np.uint8), cv2.COLORMAP_RAINBOW)
                filename_rgb_3_a = os.path.join(self.config.test_folder, name[:-4] + '_rgb_3_a.png')
                cv2.imwrite(filename_rgb_3_a, multi_fuse_rgb_3_a)

                rgb_4_a = F.interpolate(rgb_4_a, tuple(im_size), mode='bilinear', align_corners=True)
                rgb_4_a = np.squeeze(torch.sigmoid(rgb_4_a)).cpu().data.numpy()
                rgb_4_a = (rgb_4_a- rgb_4_a.min()) / (rgb_4_a.max() - rgb_4_a.min() + 1e-8)
                multi_fuse_rgb_4_a = 255 * rgb_4_a
                #multi_fuse_rgb_4_a=cv2.applyColorMap((multi_fuse_rgb_4_a).astype(np.uint8), cv2.COLORMAP_RAINBOW)
                filename_rgb_4_a = os.path.join(self.config.test_folder, name[:-4] + '_rgb_4_a.png')
                cv2.imwrite(filename_rgb_4_a, multi_fuse_rgb_4_a)

                rgb_5_a = F.interpolate(rgb_5_a, tuple(im_size), mode='bilinear', align_corners=True)
                rgb_5_a = np.squeeze(torch.sigmoid(rgb_5_a)).cpu().data.numpy()
                rgb_5_a = (rgb_5_a- rgb_5_a.min()) / (rgb_5_a.max() - rgb_5_a.min() + 1e-8)
                multi_fuse_rgb_5_a = 255 * rgb_5_a
                #multi_fuse_rgb_5_a=cv2.applyColorMap((multi_fuse_rgb_5_a).astype(np.uint8), cv2.COLORMAP_RAINBOW)
                filename_rgb_5_a = os.path.join(self.config.test_folder, name[:-4] + '_rgb_5_a.png')
                cv2.imwrite(filename_rgb_5_a, multi_fuse_rgb_5_a)

                depth_1_a = F.interpolate(depth_1_a, tuple(im_size), mode='bilinear', align_corners=True)
                depth_1_a = np.squeeze(torch.sigmoid(depth_1_a)).cpu().data.numpy()
                depth_1_a = (depth_1_a- depth_1_a.min()) / (depth_1_a.max() - depth_1_a.min() + 1e-8)
                multi_fuse_depth_1_a = 255 * depth_1_a
                #multi_fuse_depth_1_a=cv2.applyColorMap((multi_fuse_depth_1_a).astype(np.uint8), cv2.COLORMAP_RAINBOW)
                filename_depth_1_a = os.path.join(self.config.test_folder, name[:-4] + '_depth_1_a.png')
                cv2.imwrite(filename_depth_1_a, multi_fuse_depth_1_a)

                depth_2_a = F.interpolate(depth_2_a, tuple(im_size), mode='bilinear', align_corners=True)
                depth_2_a = np.squeeze(torch.sigmoid(depth_2_a)).cpu().data.numpy()
                depth_2_a = (depth_2_a- depth_2_a.min()) / (depth_2_a.max() - depth_2_a.min() + 1e-8)
                multi_fuse_depth_2_a = 255 * depth_2_a
                #multi_fuse_depth_2_a=cv2.applyColorMap((multi_fuse_depth_2_a).astype(np.uint8), cv2.COLORMAP_RAINBOW)
                filename_depth_2_a = os.path.join(self.config.test_folder, name[:-4] + '_depth_2_a.png')
                cv2.imwrite(filename_depth_2_a, multi_fuse_depth_2_a)

                depth_3_a = F.interpolate(depth_3_a, tuple(im_size), mode='bilinear', align_corners=True)
                depth_3_a = np.squeeze(torch.sigmoid(depth_3_a)).cpu().data.numpy()
                depth_3_a = (depth_3_a- depth_3_a.min()) / (depth_3_a.max() - depth_3_a.min() + 1e-8)
                multi_fuse_depth_3_a = 255 * depth_3_a
                #multi_fuse_depth_3_a=cv2.applyColorMap((multi_fuse_depth_3_a).astype(np.uint8), cv2.COLORMAP_RAINBOW)
                filename_depth_3_a = os.path.join(self.config.test_folder, name[:-4] + '_depth_3_a.png')
                cv2.imwrite(filename_depth_3_a, multi_fuse_depth_3_a)

                depth_4_a = F.interpolate(depth_4_a, tuple(im_size), mode='bilinear', align_corners=True)
                depth_4_a = np.squeeze(torch.sigmoid(depth_4_a)).cpu().data.numpy()
                depth_4_a = (depth_4_a- depth_4_a.min()) / (depth_4_a.max() - depth_4_a.min() + 1e-8)
                multi_fuse_depth_4_a = 255 * depth_4_a
                #multi_fuse_depth_4_a=cv2.applyColorMap((multi_fuse_depth_4_a).astype(np.uint8), cv2.COLORMAP_RAINBOW)
                filename_depth_4_a = os.path.join(self.config.test_folder, name[:-4] + '_depth_4_a.png')
                cv2.imwrite(filename_depth_4_a, multi_fuse_depth_4_a)

                depth_5_a = F.interpolate(depth_5_a, tuple(im_size), mode='bilinear', align_corners=True)
                depth_5_a = np.squeeze(torch.sigmoid(depth_5_a)).cpu().data.numpy()
                depth_5_a = (depth_5_a- depth_5_a.min()) / (depth_5_a.max() - depth_5_a.min() + 1e-8)
                multi_fuse_depth_5_a = 255 * depth_5_a
                #multi_fuse_depth_5_a=cv2.applyColorMap((multi_fuse_depth_5_a).astype(np.uint8), cv2.COLORMAP_RAINBOW)
                filename_depth_5_a = os.path.join(self.config.test_folder, name[:-4] + '_depth_5_a.png')
                cv2.imwrite(filename_depth_5_a, multi_fuse_depth_5_a)

                rgbd_fusion_1_a = F.interpolate(rgbd_fusion_1_a, tuple(im_size), mode='bilinear', align_corners=True)
                rgbd_fusion_1_a = np.squeeze(torch.sigmoid(rgbd_fusion_1_a)).cpu().data.numpy()
                rgbd_fusion_1_a= (rgbd_fusion_1_a- rgbd_fusion_1_a.min()) / (rgbd_fusion_1_a.max() - rgbd_fusion_1_a.min() + 1e-8)
                multi_fuse_rgbd_fusion_1_a = 255 * rgbd_fusion_1_a
                #multi_fuse_rgbd_fusion_1_a=cv2.applyColorMap((multi_fuse_rgbd_fusion_1_a).astype(np.uint8), cv2.COLORMAP_RAINBOW)
                filename_rgbd_fusion_1_a = os.path.join(self.config.test_folder, name[:-4] + '_rgbd_fusion_1_a.png')
                cv2.imwrite(filename_rgbd_fusion_1_a, multi_fuse_rgbd_fusion_1_a)

                rgbd_fusion_2_a = F.interpolate(rgbd_fusion_2_a, tuple(im_size), mode='bilinear', align_corners=True)
                rgbd_fusion_2_a = np.squeeze(torch.sigmoid(rgbd_fusion_2_a)).cpu().data.numpy()
                rgbd_fusion_2_a= (rgbd_fusion_2_a- rgbd_fusion_2_a.min()) / (rgbd_fusion_2_a.max() - rgbd_fusion_2_a.min() + 1e-8)
                multi_fuse_rgbd_fusion_2_a = 255 * rgbd_fusion_2_a
                #multi_fuse_rgbd_fusion_2_a=cv2.applyColorMap((multi_fuse_rgbd_fusion_2_a).astype(np.uint8), cv2.COLORMAP_RAINBOW)
                filename_rgbd_fusion_2_a = os.path.join(self.config.test_folder, name[:-4] + '_rgbd_fusion_2_a.png')
                cv2.imwrite(filename_rgbd_fusion_2_a, multi_fuse_rgbd_fusion_2_a)

                rgbd_fusion_3_a = F.interpolate(rgbd_fusion_3_a, tuple(im_size), mode='bilinear', align_corners=True)
                rgbd_fusion_3_a = np.squeeze(torch.sigmoid(rgbd_fusion_3_a)).cpu().data.numpy()
                rgbd_fusion_3_a= (rgbd_fusion_3_a- rgbd_fusion_3_a.min()) / (rgbd_fusion_3_a.max() - rgbd_fusion_3_a.min() + 1e-8)
                multi_fuse_rgbd_fusion_3_a = 255 * rgbd_fusion_3_a
                #multi_fuse_rgbd_fusion_3_a=cv2.applyColorMap((multi_fuse_rgbd_fusion_3_a).astype(np.uint8), cv2.COLORMAP_RAINBOW)
                filename_rgbd_fusion_3_a = os.path.join(self.config.test_folder, name[:-4] + '_rgbd_fusion_3_a.png')
                cv2.imwrite(filename_rgbd_fusion_3_a, multi_fuse_rgbd_fusion_3_a)

                rgbd_fusion_4_a = F.interpolate(rgbd_fusion_4_a, tuple(im_size), mode='bilinear', align_corners=True)
                rgbd_fusion_4_a = np.squeeze(torch.sigmoid(rgbd_fusion_4_a)).cpu().data.numpy()
                rgbd_fusion_4_a= (rgbd_fusion_4_a- rgbd_fusion_4_a.min()) / (rgbd_fusion_4_a.max() - rgbd_fusion_4_a.min() + 1e-8)
                multi_fuse_rgbd_fusion_4_a = 255 * rgbd_fusion_4_a
                #multi_fuse_rgbd_fusion_4_a=cv2.applyColorMap((multi_fuse_rgbd_fusion_4_a).astype(np.uint8), cv2.COLORMAP_RAINBOW)
                filename_rgbd_fusion_4_a = os.path.join(self.config.test_folder, name[:-4] + '_rgbd_fusion_4_a.png')
                cv2.imwrite(filename_rgbd_fusion_4_a, multi_fuse_rgbd_fusion_4_a)

                rgbd_fusion_5_a = F.interpolate(rgbd_fusion_5_a, tuple(im_size), mode='bilinear', align_corners=True)
                rgbd_fusion_5_a = np.squeeze(torch.sigmoid(rgbd_fusion_5_a)).cpu().data.numpy()
                rgbd_fusion_5_a= (rgbd_fusion_5_a- rgbd_fusion_5_a.min()) / (rgbd_fusion_5_a.max() - rgbd_fusion_5_a.min() + 1e-8)
                multi_fuse_rgbd_fusion_5_a = 255 * rgbd_fusion_5_a
                #multi_fuse_rgbd_fusion_5_a=cv2.applyColorMap((multi_fuse_rgbd_fusion_5_a).astype(np.uint8), cv2.COLORMAP_RAINBOW)
                filename_rgbd_fusion_5_a = os.path.join(self.config.test_folder, name[:-4] + '_rgbd_fusion_5_a.png')
                cv2.imwrite(filename_rgbd_fusion_5_a, multi_fuse_rgbd_fusion_5_a)
                #print(rgb_1_a.shape)
                model_weights =[]
                conv_layers = []# get all the model children as list
                model_children = list(self.net.modules())#counter to keep count of the conv layers
                counter = 0#append all the conv layers and their respective wights to the list
                for i in range(len(model_children)):
                    print('name of model childgren',model_children[i])
                    if type(model_children[i]) == nn.Conv2d:
                        counter+=1
                        model_weights.append(model_children[i].weight)
                        conv_layers.append(model_children[i])
                    elif type(model_children[i]) == nn.Sequential:
                        for j in range(len(model_children[i])):
                            for child in model_children[i][j].children():
                                if type(child) == nn.Conv2d:
                                    counter+=1
                                    model_weights.append(child.weight)
                                    conv_layers.append(child)
                print(f"Total sequential layers: {counter}")
                print(conv_layers)


                outputs = []
                names = []
                for layer in conv_layers[0:]:
                    image = layer(images)
                    outputs.append(image)
                    names.append(str(layer))
                print(len(outputs))#print feature_maps
                for feature_map in outputs:
                    print(feature_map.shape)

                #print(e_rgbd01.shape)
                #preds = F.interpolate(preds, tuple(im_size), mode='bilinear', align_corners=True)
                pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()
                print(pred.shape)
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                multi_fuse = 255 * pred
                filename = os.path.join(self.config.test_folder, name[:-4] + '_convtran.png')
                cv2.imwrite(filename, multi_fuse)
                #coarse_sal_rgb= F.interpolate(coarse_sal_rgb, tuple(im_size), mode='bilinear', align_corners=True)
                coarse_sal_rgbs = np.squeeze(torch.sigmoid(coarse_sal_rgb)).cpu().data.numpy()
                #print(pred.shape)
                coarse_sal_rgbs = (coarse_sal_rgbs - coarse_sal_rgbs.min()) / (coarse_sal_rgbs.max() - coarse_sal_rgbs.min() + 1e-8)
                multi_fuse_coarse_sal_rgb = 255 * coarse_sal_rgbs
                filename_r = os.path.join(self.config.test_folder, name[:-4] + '_coarse_sal_rgb.png')
                cv2.imwrite(filename_r, multi_fuse_coarse_sal_rgb)
                coarse_sal_depth= F.interpolate(coarse_sal_depth, tuple(im_size), mode='bilinear', align_corners=True)
                coarse_sal_ds = np.squeeze(torch.sigmoid(coarse_sal_depth)).cpu().data.numpy()
                #print(pred.shape)
                coarse_sal_ds = (coarse_sal_ds - coarse_sal_ds.min()) / (coarse_sal_ds.max() - coarse_sal_ds.min() + 1e-8)
                multi_fuse_coarse_sal_ds = 255 * coarse_sal_ds
                filename_d = os.path.join(self.config.test_folder, name[:-4] + '_coarse_sal_d.png')
                cv2.imwrite(filename_d, multi_fuse_coarse_sal_ds)
                e_rgbd2= F.interpolate(e_rgbd2, tuple(im_size), mode='bilinear', align_corners=True)
                e_rgbd2 = np.squeeze(torch.sigmoid(e_rgbd0)).cpu().data.numpy()
                #print(pred.shape)
                e_rgbd2 = (e_rgbd2 - e_rgbd2.min()) / (e_rgbd2.max() - e_rgbd2.min() + 1e-8)
                multi_fuse_e_rgbd2 = 255 * e_rgbd2
                filename_re = os.path.join(self.config.test_folder, name[:-4] + '_edge2.png')
                cv2.imwrite(filename_re, multi_fuse_e_rgbd2)

               
                #e_rgbd01 = F.interpolate(e_rgbd01, tuple(im_size), mode='bilinear', align_corners=True)
                e_rgbd01 = np.squeeze(torch.sigmoid(Att[10])).cpu().data.numpy()
                print(e_rgbd01.shape)
                #e_rgbd01 = (e_rgbd01-e_rgbd01.min()) / (e_rgbd01.max() - e_rgbd01.min() + 1e-8)
                #e_rgbd01 = 255 * e_rgbd01
                filename = os.path.join(self.config.test_folder, name[:-4] + '_edge.png')
                a=cv2.imwrite(filename, e_rgbd01)
                print(a)
                #e_rgbd11 = F.interpolate(e_rgbd11, tuple(im_size), mode='bilinear', align_corners=True)
                e_rgbd11 = np.squeeze(torch.sigmoid(Att[11])).cpu().data.numpy()
                print(e_rgbd11.shape)
                e_rgbd11 = (e_rgbd11-e_rgbd11.min()) / (e_rgbd11.max() - e_rgbd11.min() + 1e-8)
                e_rgbd11 = 255 * e_rgbd11
                filename = os.path.join(self.config.test_folder, name[:-5] + '_edge.png')
                cv2.imwrite(filename, e_rgbd11)

                #e_rgbd21 = F.interpolate(e_rgbd21, tuple(im_size), mode='bilinear', align_corners=True)
                e_rgbd21 = np.squeeze(torch.sigmoid(Att[9])).cpu().data.numpy()

                e_rgbd21 = (e_rgbd21-e_rgbd21.min()) / (e_rgbd21.max() - e_rgbd21.min() + 1e-8)
                e_rgbd21 = 255 * e_rgbd21
                filename = os.path.join(self.config.test_folder, name[:-6] + '_edge.png')
                cv2.imwrite(filename, e_rgbd21)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')
    
  
    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        
        loss_vals=  []
        
        for epoch in range(self.config.epoch):
            r_sal_loss = 0
            r_sal_loss_item=0
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_depth, sal_label, sal_edge = data_batch['sal_image'], data_batch['sal_depth'], data_batch[
                    'sal_label'], data_batch['sal_edge']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    sal_image, sal_depth, sal_label, sal_edge= sal_image.to(device), sal_depth.to(device), sal_label.to(device),sal_edge.to(device)

               
                self.optimizer.zero_grad()
                sal_label_coarse = F.interpolate(sal_label, size_coarse, mode='bilinear', align_corners=True)
                
                sal_final,coarse_sal_rgb,sal_edge_rgbd0 = self.net(sal_image)
                
                sal_loss_coarse_rgb =  F.binary_cross_entropy_with_logits(coarse_sal_rgb, sal_label_coarse, reduction='sum')
                
                sal_final_loss =  F.binary_cross_entropy_with_logits(sal_final, sal_label, reduction='sum')
                edge_loss_rgbd0=F.smooth_l1_loss(sal_edge_rgbd0,sal_edge)
              
                
                sal_loss_fuse = sal_final_loss+edge_loss_rgbd0+sal_loss_coarse_rgb
                sal_loss = sal_loss_fuse/ (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.data
                r_sal_loss_item+=sal_loss.item() * sal_image.size(0)
                sal_loss.backward()
                self.optimizer.step()

                if (i + 1) % (self.show_every // self.config.batch_size) == 0:
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %0.4f  ||sal_final:%0.4f|| edge_loss0:%0.4f' % (
                        epoch, self.config.epoch, i + 1, iter_num, r_sal_loss,sal_final_loss,edge_loss_rgbd0 ))
                    # print('Learning rate: ' + str(self.lr))
                    writer.add_scalar('training loss', r_sal_loss / (self.show_every / self.iter_size),
                                      epoch * len(self.train_loader.dataset) + i)
                    writer.add_scalar('sal_loss_coarse_rgb training loss', sal_loss_coarse_rgb.data,
                                      epoch * len(self.train_loader.dataset) + i)
                    writer.add_scalar('sal_loss_coarse_depth training loss', sal_loss_coarse_depth.data,
                                      epoch * len(self.train_loader.dataset) + i)
                    writer.add_scalar('sal_edge training loss', edge_loss_rgbd2.data,
                                      epoch * len(self.train_loader.dataset) + i)

                    r_sal_loss = 0
                    res = coarse_sal_depth[0].clone()
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    writer.add_image('coarse_sal_depth', torch.tensor(res), i, dataformats='HW')
                    grid_image = make_grid(sal_label_coarse[0].clone().cpu().data, 1, normalize=True)

                    res = coarse_sal_rgb[0].clone()
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    writer.add_image('coarse_sal_rgb', torch.tensor(res), i, dataformats='HW')
                    grid_image = make_grid(sal_label_coarse[0].clone().cpu().data, 1, normalize=True)
                    
                    fsal = sal_final[0].clone()
                    fsal = fsal.sigmoid().data.cpu().numpy().squeeze()
                    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min() + 1e-8)
                    writer.add_image('sal_final', torch.tensor(fsal), i, dataformats='HW')
                    grid_image = make_grid(sal_label[0].clone().cpu().data, 1, normalize=True)

                    fsal = sal_low[0].clone()
                    fsal = fsal.sigmoid().data.cpu().numpy().squeeze()
                    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min() + 1e-8)
                    writer.add_image('sal_low', torch.tensor(fsal), i, dataformats='HW')
                    grid_image = make_grid(sal_label[0].clone().cpu().data, 1, normalize=True)
                    fsal = sal_high[0].clone()
                    fsal = fsal.sigmoid().data.cpu().numpy().squeeze()
                    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min() + 1e-8)
                    writer.add_image('sal_high', torch.tensor(fsal), i, dataformats='HW')
                    grid_image = make_grid(sal_label[0].clone().cpu().data, 1, normalize=True)
                    fsal = sal_med[0].clone()
                    fsal = fsal.sigmoid().data.cpu().numpy().squeeze()
                    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min() + 1e-8)
                    writer.add_image('sal_med', torch.tensor(fsal), i, dataformats='HW')
                    grid_image = make_grid(sal_label[0].clone().cpu().data, 1, normalize=True)
                    sal_edge_rgbd = sal_edge_rgbd2[0].clone()
                    sal_edge_rgbd = sal_edge_rgbd.sigmoid().data.cpu().numpy().squeeze()
                    sal_edge_rgbd = (sal_edge_rgbd - sal_edge_rgbd.min()) / (sal_edge_rgbd.max() - sal_edge_rgbd.min() + 1e-8)
                    writer.add_image('sal_edge_rgbd', torch.tensor(sal_edge_rgbd), i, dataformats='HW')
                    grid_image = make_grid(sal_edge[0].clone().cpu().data, 1, normalize=True)


            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))
            train_loss=r_sal_loss_item/len(self.train_loader.dataset)
            loss_vals.append(train_loss)
            
            print('Epoch:[%2d/%2d] | Train Loss : %.3f' % (epoch, self.config.epoch,train_loss))
            
        # save model
        torch.save(self.net.state_dict(), '%s/final.pth' % self.config.save_folder)'''
import torch
from torch.nn import functional as F
from conformer import build_model
import numpy as np
import os
import cv2
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch.nn as nn
import argparse
import os.path as osp
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import matplotlib.pyplot as plt
import scipy.misc
from utils import count_model_flops, count_model_params
from PIL import Image
import json

class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        
        # Build and script the model
        self.net = build_model(self.config.network, self.config.arch)

        if config.mode == 'test':
            print(f'Loading pre-trained model from {self.config.model}...')
            self.net.load_state_dict(torch.load(self.config.model, map_location=torch.device('cpu')))
        
        if self.config.cuda:
            self.net = self.net.cuda()
        
        # Convert to TorchScript for faster inference
        self.scripted_net = torch.jit.script(self.net)
        
        self.lr = self.config.lr
        self.wd = self.config.wd
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        self.print_network(self.net, 'Conformer based SOD Structure')

    def print_network(self, model, name):
        num_params_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_params = sum(p.numel() for p in model.parameters())
        print(name)
        print(f'Trainable parameters: {num_params_t}')
        print(f'Total parameters: {num_params}')
        print(f'FLOPs: {count_model_flops(model)}')
        print(f'Params: {count_model_params(model)}')

    def test(self):
        print('Testing...')
        time_s = time.time()
        img_num = len(self.test_loader)
        
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size, depth = (
                data_batch['image'],
                data_batch['name'][0],
                np.asarray(data_batch['size']),
                data_batch['depth']
            )
            
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device)
                    depth = depth.to(device)
                if self.config.cuda:
                    torch.cuda.synchronize()

                start_time = time.time()  # Timing starts AFTER synchronization
                preds, _, _, _, _, _ = self.scripted_net(images)
                if self.config.cuda:
                    torch.cuda.synchronize()

                frame_time = time.time() - start_time  # Time for one frame
                print(frame_time)
                pred = torch.sigmoid(preds).cpu().numpy().squeeze()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                multi_fuse = (255 * pred).astype(np.uint8)
                filename = os.path.join(self.config.test_folder, name[:-4] + '_convtran.png')
                cv2.imwrite(filename, multi_fuse)
        
        time_e = time.time()
        print(f'Speed: {img_num / (time_e - time_s):.2f} FPS')
        
        print('Test Done!')
    
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        loss_vals = []
        
        for epoch in range(self.config.epoch):
            r_sal_loss_item = 0
            
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_depth, sal_label, sal_edge = (
                    data_batch['sal_image'],
                    data_batch['sal_depth'],
                    data_batch['sal_label'],
                    data_batch['sal_edge']
                )
                
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    sal_image, sal_depth, sal_label, sal_edge = (
                        sal_image.to(device),
                        sal_depth.to(device),
                        sal_label.to(device),
                        sal_edge.to(device)
                    )
                
                self.optimizer.zero_grad()
                sal_label_coarse = F.interpolate(sal_label, (10, 10), mode='bilinear', align_corners=True)
                sal_final, coarse_sal_rgb, sal_edge_rgbd0 = self.net(sal_image)
                
                sal_loss_coarse_rgb = F.binary_cross_entropy_with_logits(coarse_sal_rgb, sal_label_coarse, reduction='sum')
                sal_final_loss = F.binary_cross_entropy_with_logits(sal_final, sal_label, reduction='sum')
                edge_loss_rgbd0 = F.smooth_l1_loss(sal_edge_rgbd0, sal_edge)
                
                sal_loss = (sal_final_loss + edge_loss_rgbd0 + sal_loss_coarse_rgb) / (self.iter_size * self.config.batch_size)
                r_sal_loss_item += sal_loss.item() * sal_image.size(0)
                
                sal_loss.backward()
                self.optimizer.step()
                
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), f'{self.config.save_folder}/epoch_{epoch + 1}.pth')
                
            train_loss = r_sal_loss_item / len(self.train_loader.dataset)
            loss_vals.append(train_loss)
            print(f'Epoch:[{epoch + 1}/{self.config.epoch}] | Train Loss: {train_loss:.3f}')
            
        torch.save(self.net.state_dict(), f'{self.config.save_folder}/final.pth')
    

