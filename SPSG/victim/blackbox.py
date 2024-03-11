#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
import json
from matplotlib import cm
import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
torch.cuda.device_count()
torch.cuda.is_available()
torch.cuda.device_count()
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lime
import random
from lime import lime_image
from SPSG.utils.type_checks import TypeCheck
import SPSG.utils.model as model_utils
import SPSG.models.zoo as zoo
from SPSG import datasets
from lime.wrappers.scikit_image import SegmentationAlgorithm



class Blackbox(object):
    def __init__(self, model, device=None, output_type='probs', topk=None, rounding=None):
        self.device = torch.device('cuda') if device is None else device
        self.output_type = output_type
        self.topk = topk
        self.rounding = rounding

        self.__model = model.to(device)
        self.output_type = output_type
        self.__model.eval()

        self.__call_count = 0

    def __call__(self, query_input):
        TypeCheck.multiple_image_blackbox_input_tensor(query_input)

        with torch.no_grad():
            query_input = query_input.to(self.device)
            query_output = self.__model(query_input)

            if isinstance(query_output, tuple):
                # In certain cases, the models additional outputs during forward pass
                # e.g., activation maps in WideResNets of Zero-shot KT
                # Restrict to just the logits -- which we assume by default is the first element
                query_output = query_output[0]

            self.__call_count += query_input.shape[0]

            query_output_probs = F.softmax(query_output, dim=1)

        query_output_probs = self.truncate_output(query_output_probs)
        return query_output_probs

    @classmethod
    def from_modeldir(cls, model_dir, device=None, output_type='probs'):
        device = torch.device('cuda') if device is None else device

        # What was the model architecture used by this model?
        params_path = osp.join(model_dir, 'params.json')
        with open(params_path) as jf:
            params = json.load(jf)
        model_arch = params['model_arch']
        num_classes = params['num_classes']
        victim_dataset = params.get('dataset', 'imagenet')
        modelfamily = datasets.dataset_to_modelfamily[victim_dataset]

        # Instantiate the model
        # model = model_utils.get_net(model_arch, n_output_classes=num_classes)
        model = zoo.get_net(model_arch, modelfamily, pretrained=None, num_classes=num_classes)
        model = model.to(device)

        # Load weights
        checkpoint_path = osp.join(model_dir, 'model_best.pth.tar')
        if not osp.exists(checkpoint_path):
            checkpoint_path = osp.join(model_dir, 'checkpoint.pth.tar')
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc))

        blackbox = cls(model, device, output_type)
        return blackbox

    def get_model(self):
        print('======================================================================================================')
        print('WARNING: USE get_model() *ONLY* FOR DEBUGGING')
        print('======================================================================================================')
        return self.__model

    def truncate_output(self, y_t_probs):
        if self.topk is not None:
            # Zero-out everything except the top-k predictions
            topk_vals, indices = torch.topk(y_t_probs, self.topk)
            newy = torch.zeros_like(y_t_probs)
            if self.rounding == 0:
                # argmax prediction
                newy = newy.scatter(1, indices, torch.ones_like(topk_vals))
            else:
                newy = newy.scatter(1, indices, topk_vals)
            y_t_probs = newy

        # Rounding of decimals
        if self.rounding is not None:
            y_t_probs = torch.Tensor(np.round(y_t_probs.numpy(), decimals=self.rounding))

        return y_t_probs

    def train(self):
        raise ValueError('Cannot run blackbox model in train mode')

    def eval(self):
        # Always in eval mode
        pass

    def get_call_count(self):
        return self.__call_count







    def color2gray(self,color_img):
        size_h, size_w, channel = color_img.shape
        gray_img = np.zeros((size_h, size_w), dtype=np.uint8)
        for i in range(size_h):
            for j in range(size_w):
                gray_img[i, j] = round((color_img[i, j, 0] * 30 + color_img[i, j, 1] * 59 + \
                                        color_img[i, j, 2] * 11) / 100)
        return gray_img

    def get_SG_distribution(self,samples):
        SGs = []
        queries = []
        for origin_sample in samples:
            sample = origin_sample.permute(1,2,0).cpu().numpy()
            # plt.imshow(sample)
            # plt.show()
            explainer = lime_image.LimeImageExplainer()
            xx = sample.astype(np.double)  # lime要求numpy array

            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=99)
            segments = segmentation_fn(xx)
            # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            # plt.show()

            sample_tensor = torch.tensor(copy.deepcopy(sample)).unsqueeze(dim=0).permute(0, 3, 1, 2)
            sample_tensor = sample_tensor.to(self.device)
            query_output2 = self.__model(sample_tensor)
            self.__call_count += sample_tensor.shape[0]

            SG = torch.zeros_like(sample_tensor)
            print(np.unique(segments))
            for x in np.unique(segments):

                if np.any(segments==x):
                    for channel in range(3):
                        sample_permutation1 = copy.deepcopy(sample)
                        # sample_permutation2 = copy.deepcopy(sample)

                        with torch.no_grad():
                            sample_permutation1 = torch.tensor(sample_permutation1).unsqueeze(dim=0).permute(0, 3, 1, 2).to(self.device)

                            sample_permutation1[0, channel][(segments == x) ] += 1e-4
                            query_input1 = sample_permutation1.to(self.device)
                            query_output1 = self.__model(query_input1)
                            self.__call_count += query_input1.shape[0]
                            # sample_permutation2 = torch.tensor(sample_permutation2).unsqueeze(dim=0).permute(0, 3, 1, 2)
                            # sample_permutation1[0, channel][torch.tensor((segments == x))] -= 1e-4
                            # query_input2 = sample_permutation2.to(self.device)

                            _, predict = torch.max(torch.log_softmax(query_output1,dim=1).squeeze(), dim=0)

                            a = (F.nll_loss(query_output1,predict.unsqueeze(dim=0)).item()-F.nll_loss(query_output2,predict.unsqueeze(dim=0)).item())/0.0001
                            SG[0, channel][torch.tensor((segments==x))]=a
            print("call_count",self.__call_count)



            # reg1 = self.calculate(SG)
            # reg11 = reg1.view(3, sample_tensor.shape[2],  sample_tensor.shape[3]).permute(2, 1, 0).cpu().numpy()
            # reg2 = self.color2gray(reg11)
            #
            #
            #
            # fig = plt.figure()
            # ax = plt.axes(projection="3d")
            # X = np.arange(0, 224, 1)
            # Y = np.arange(0, 224, 1)
            # X, Y = np.meshgrid(X, Y)
            # surf = ax.plot_surface(X, Y, reg2, cmap=cm.coolwarm,
            #                        linewidth=0, antialiased=False)
            #
            # ax.set_zlim(0, 255)  # z轴的取值范围
            # # ax.zaxis.set_major_locator(LinearLocator(10))
            # # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            #
            # fig.colorbar(surf, shrink=0.5, aspect=5)
            # plt.show()
            # a = reg1.nonzero(as_tuple = False)
            # b = reg1[reg1.nonzero(as_tuple = True)]
            # print(a,b)
            # SGs.append([a,b])
            SGs.append(SG)
            queries.append(query_output2)
        return SGs,queries






