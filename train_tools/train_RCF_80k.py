import os
import numpy as np
import random
import torch
seed=1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from tqdm import tqdm
from utils.lr_scheduler import PolyLrUpdaterHook
from dataloaders.datasets.bsds_hd5 import Mydataset
from torch.utils.data import DataLoader
from my_options.RCF_options import RCF_Options
from modeling.RCF import RCF
import torch.nn as nn
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.rcf_functions import  cross_entropy_loss_RCF, SGD_caffe, cross_entropy_loss_RCF2
from utils.rcf_utils import Logger, Averagvalue, save_checkpoint, load_vgg16pretrain
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
import scipy.io as sio
import time
from utils.log import get_logger
import cv2

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()


        print(self.saver.experiment_dir)
        self.output_dir = os.path.join(self.saver.experiment_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.logger = get_logger(self.output_dir+'/log.txt')
        self.logger.info('*' * 80)
        self.logger.info('the args are the below')
        self.logger.info('*' * 80)
        for x in self.args.__dict__:
            self.logger.info(x + ',' + str(self.args.__dict__[x]))
        self.logger.info('*' * 80)

        # Define Dataloader
        self.train_dataset = Mydataset(root_path=self.args.data_path, split='trainval', crop_size=self.args.crop_size)
        self.test_dataset = Mydataset(root_path=self.args.data_path, split='test', crop_size=self.args.crop_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                       num_workers=args.workers, pin_memory=True, drop_last=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                                      num_workers=args.workers)
        # Define network
        self.model = RCF()
        self.model.cuda()
        # Using cuda
        # if self.args.cuda:
        #    self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
        #    patch_replication_callback(self.model)
        #    self.model = self.model.cuda()

        self.model.apply(weights_init)
        load_vgg16pretrain(self.model, self.args.pretrain_model)
        if args.resume:
            if isfile(args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}'"
                      .format(args.resume))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        # tune lr
        net_parameters_id = {}
        net = self.model
        for pname, p in net.named_parameters():
            if pname in ['conv1_1.weight', 'conv1_2.weight',
                         'conv2_1.weight', 'conv2_2.weight',
                         'conv3_1.weight', 'conv3_2.weight', 'conv3_3.weight',
                         'conv4_1.weight', 'conv4_2.weight', 'conv4_3.weight']:
                print(pname, 'lr:1 de:1')
                if 'conv1-4.weight' not in net_parameters_id:
                    net_parameters_id['conv1-4.weight'] = []
                net_parameters_id['conv1-4.weight'].append(p)
            elif pname in ['conv1_1.bias', 'conv1_2.bias',
                           'conv2_1.bias', 'conv2_2.bias',
                           'conv3_1.bias', 'conv3_2.bias', 'conv3_3.bias',
                           'conv4_1.bias', 'conv4_2.bias', 'conv4_3.bias']:
                print(pname, 'lr:2 de:0')
                if 'conv1-4.bias' not in net_parameters_id:
                    net_parameters_id['conv1-4.bias'] = []
                net_parameters_id['conv1-4.bias'].append(p)
            elif pname in ['conv5_1.weight', 'conv5_2.weight', 'conv5_3.weight']:
                print(pname, 'lr:100 de:1')
                if 'conv5.weight' not in net_parameters_id:
                    net_parameters_id['conv5.weight'] = []
                net_parameters_id['conv5.weight'].append(p)
            elif pname in ['conv5_1.bias', 'conv5_2.bias', 'conv5_3.bias']:
                print(pname, 'lr:200 de:0')
                if 'conv5.bias' not in net_parameters_id:
                    net_parameters_id['conv5.bias'] = []
                net_parameters_id['conv5.bias'].append(p)
            elif pname in ['conv1_1_down.weight', 'conv1_2_down.weight',
                           'conv2_1_down.weight', 'conv2_2_down.weight',
                           'conv3_1_down.weight', 'conv3_2_down.weight', 'conv3_3_down.weight',
                           'conv4_1_down.weight', 'conv4_2_down.weight', 'conv4_3_down.weight',
                           'conv5_1_down.weight', 'conv5_2_down.weight', 'conv5_3_down.weight']:
                print(pname, 'lr:0.1 de:1')
                if 'conv_down_1-5.weight' not in net_parameters_id:
                    net_parameters_id['conv_down_1-5.weight'] = []
                net_parameters_id['conv_down_1-5.weight'].append(p)
            elif pname in ['conv1_1_down.bias', 'conv1_2_down.bias',
                           'conv2_1_down.bias', 'conv2_2_down.bias',
                           'conv3_1_down.bias', 'conv3_2_down.bias', 'conv3_3_down.bias',
                           'conv4_1_down.bias', 'conv4_2_down.bias', 'conv4_3_down.bias',
                           'conv5_1_down.bias', 'conv5_2_down.bias', 'conv5_3_down.bias']:
                print(pname, 'lr:0.2 de:0')
                if 'conv_down_1-5.bias' not in net_parameters_id:
                    net_parameters_id['conv_down_1-5.bias'] = []
                net_parameters_id['conv_down_1-5.bias'].append(p)
            elif pname in ['score_dsn1.weight', 'score_dsn2.weight', 'score_dsn3.weight',
                           'score_dsn4.weight', 'score_dsn5.weight']:
                print(pname, 'lr:0.01 de:1')
                if 'score_dsn_1-5.weight' not in net_parameters_id:
                    net_parameters_id['score_dsn_1-5.weight'] = []
                net_parameters_id['score_dsn_1-5.weight'].append(p)
            elif pname in ['score_dsn1.bias', 'score_dsn2.bias', 'score_dsn3.bias',
                           'score_dsn4.bias', 'score_dsn5.bias']:
                print(pname, 'lr:0.02 de:0')
                if 'score_dsn_1-5.bias' not in net_parameters_id:
                    net_parameters_id['score_dsn_1-5.bias'] = []
                net_parameters_id['score_dsn_1-5.bias'].append(p)
            elif pname in ['score_final_depth.weight']:
                print(pname, 'lr:0.001 de:1')
                if 'score_final_depth.weight' not in net_parameters_id:
                    net_parameters_id['score_final_depth.weight'] = []
                net_parameters_id['score_final_depth.weight'].append(p)
            elif pname in ['score_final_normal.weight']:
                print(pname, 'lr:0.001 de:1')
                if 'score_final_normal.weight' not in net_parameters_id:
                    net_parameters_id['score_final_normal.weight'] = []
                net_parameters_id['score_final_normal.weight'].append(p)
            elif pname in ['score_final_reflectance.weight']:
                print(pname, 'lr:0.001 de:1')
                if 'score_final_reflectance.weight' not in net_parameters_id:
                    net_parameters_id['score_final_reflectance.weight'] = []
                net_parameters_id['score_final_reflectance.weight'].append(p)
            elif pname in ['score_final_illumination.weight']:
                print(pname, 'lr:0.001 de:1')
                if 'score_final_illumination.weight' not in net_parameters_id:
                    net_parameters_id['score_final_illumination.weight'] = []
                net_parameters_id['score_final_illumination.weight'].append(p)
            elif pname in ['score_final_depth.bias']:
                print(pname, 'lr:0.002 de:0')
                if 'score_final_depth.bias' not in net_parameters_id:
                    net_parameters_id['score_final_depth.bias'] = []
                net_parameters_id['score_final_depth.bias'].append(p)
            elif pname in ['score_final_normal.bias']:
                print(pname, 'lr:0.002 de:0')
                if 'score_final_normal.bias' not in net_parameters_id:
                    net_parameters_id['score_final_normal.bias'] = []
                net_parameters_id['score_final_normal.bias'].append(p)
            elif pname in ['score_final_reflectance.bias']:
                print(pname, 'lr:0.002 de:0')
                if 'score_final_reflectance.bias' not in net_parameters_id:
                    net_parameters_id['score_final_reflectance.bias'] = []
                net_parameters_id['score_final_reflectance.bias'].append(p)
            elif pname in ['score_final_illumination.bias']:
                print(pname, 'lr:0.002 de:0')
                if 'score_final_illumination.bias' not in net_parameters_id:
                    net_parameters_id['score_final_illumination.bias'] = []
                net_parameters_id['score_final_illumination.bias'].append(p)

        self.optimizer = torch.optim.SGD([
            {'params': net_parameters_id['conv1-4.weight'], 'lr': self.args.lr * 1,
             'weight_decay': self.args.weight_decay},
            {'params': net_parameters_id['conv1-4.bias'], 'lr': self.args.lr * 2, 'weight_decay': 0.},
            {'params': net_parameters_id['conv5.weight'], 'lr': self.args.lr * 100,
             'weight_decay': self.args.weight_decay},
            {'params': net_parameters_id['conv5.bias'], 'lr': self.args.lr * 200, 'weight_decay': 0.},
            {'params': net_parameters_id['conv_down_1-5.weight'], 'lr': self.args.lr * 0.1,
             'weight_decay': self.args.weight_decay},
            {'params': net_parameters_id['conv_down_1-5.bias'], 'lr': self.args.lr * 0.2, 'weight_decay': 0.},
            {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': self.args.lr * 0.01,
             'weight_decay': self.args.weight_decay},
            {'params': net_parameters_id['score_dsn_1-5.bias'], 'lr': self.args.lr * 0.02, 'weight_decay': 0.},
            {'params': net_parameters_id['score_final_depth.weight'], 'lr': self.args.lr * 0.001,
             'weight_decay': self.args.weight_decay},
            {'params': net_parameters_id['score_final_normal.weight'], 'lr': self.args.lr * 0.001,
             'weight_decay': self.args.weight_decay},
            {'params': net_parameters_id['score_final_reflectance.weight'], 'lr': self.args.lr * 0.001,
             'weight_decay': self.args.weight_decay},
            {'params': net_parameters_id['score_final_illumination.weight'], 'lr': self.args.lr * 0.001,
             'weight_decay': self.args.weight_decay},
            {'params': net_parameters_id['score_final_depth.bias'], 'lr': self.args.lr * 0.002, 'weight_decay': 0.},
            {'params': net_parameters_id['score_final_normal.bias'], 'lr': self.args.lr * 0.002, 'weight_decay': 0.},
            {'params': net_parameters_id['score_final_reflectance.bias'], 'lr': self.args.lr * 0.002, 'weight_decay': 0.},
            {'params': net_parameters_id['score_final_illumination.bias'], 'lr': self.args.lr * 0.002, 'weight_decay': 0.},
        ], lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        # Define lr scheduler
        self.scheduler = PolyLrUpdaterHook(power=0.9, base_lr=self.args.lr, min_lr=self.args.minlr)

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    def training(self):
        cur = 0
        data_iter = iter(self.train_loader)
        iter_per_epoch = len(self.train_loader)
        self.logger.info('*' * 40)
        self.logger.info('train images in all are %d ' % (iter_per_epoch*self.args.batch_size))
        self.logger.info('*' * 40)

        train_loss = 0.0
        self.model.train()
        start_time = time.time()
        for step in range(self.args.start_iters, self.args.total_iters):

            if cur == iter_per_epoch:
                cur = 0
                data_iter = iter(self.train_loader)
            image, target = next(data_iter)
            if self.args.cuda:
                image, target = image.cuda(), target.cuda() #(b,3,w,h) (b,1,w,h)
                target = target[:, 1:5, :, :]

            outputs = self.model(image)
            loss = torch.zeros(1).cuda()
            for o in outputs:
                loss = loss + cross_entropy_loss_RCF2(o, target)

            self.scheduler(self.optimizer, step, self.args.total_iters)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            if (step+1) % self.args.snapshots == 0:
                self.saver.save_checkpoint({
                    'epoch': step + 1, 'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(), 'best_pred': self.best_pred,
                }, is_best=False)
                self.test(step)
                self.multiscale_test(step)
                self.model.train()

            if (step+1) % self.args.display == 0:
                tm = time.time() - start_time
                self.logger.info('iter: %d, lr: %e, loss: %f, time using: %f(%fs/iter)'
                                 % ((step+1), self.optimizer.param_groups[0]['lr'], (train_loss / (step + 1)), tm, tm / self.args.display))
                start_time = time.time()

            cur = cur+1

        print('Loss: %.3f' % train_loss)


    def test(self, iters):
        print('Test epoch: %d' % iters)
        self.output_dir = os.path.join(self.saver.experiment_dir, str(iters+1))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.depth_output_dir = os.path.join(self.output_dir, 'depth/mat')
        if not os.path.exists(self.depth_output_dir):
            os.makedirs(self.depth_output_dir)
        self.normal_output_dir = os.path.join(self.output_dir, 'normal/mat')
        if not os.path.exists(self.normal_output_dir):
            os.makedirs(self.normal_output_dir)
        self.reflectance_output_dir = os.path.join(self.output_dir, 'reflectance/mat')
        if not os.path.exists(self.reflectance_output_dir):
            os.makedirs(self.reflectance_output_dir)
        self.illumination_output_dir = os.path.join(self.output_dir, 'illumination/mat')
        if not os.path.exists(self.illumination_output_dir):
            os.makedirs(self.illumination_output_dir)

        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        for i, image in enumerate(tbar):
            name = self.test_loader.dataset.images_name[i]
            if self.args.cuda:
                image = image.cuda()
            with torch.no_grad():
                output_list = self.model(image)

            pred = output_list[-1]
            pred = pred.squeeze()
            out_depth = pred[0, :, :]
            out_normal = pred[1, :, :]
            out_reflectance = pred[2, :, :]
            out_illumination = pred[3, :, :]

            depth_pred = out_depth.data.cpu().numpy()
            depth_pred = depth_pred.squeeze()
            sio.savemat(os.path.join(self.depth_output_dir, '{}.mat'.format(name)), {'result': depth_pred})

            normal_pred = out_normal.data.cpu().numpy()
            normal_pred = normal_pred.squeeze()
            sio.savemat(os.path.join(self.normal_output_dir, '{}.mat'.format(name)), {'result': normal_pred})

            reflectance_pred = out_reflectance.data.cpu().numpy()
            reflectance_pred = reflectance_pred.squeeze()
            sio.savemat(os.path.join(self.reflectance_output_dir, '{}.mat'.format(name)), {'result': reflectance_pred})

            illumination_pred = out_illumination.data.cpu().numpy()
            illumination_pred = illumination_pred.squeeze()
            sio.savemat(os.path.join(self.illumination_output_dir, '{}.mat'.format(name)),
                        {'result': illumination_pred})

    def multiscale_test(self, iters):
        print('Test epoch: %d' % iters)
        self.output_dir = os.path.join(self.saver.experiment_dir, str(iters + 1) + '_ms')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.depth_output_dir = os.path.join(self.output_dir, 'depth/mat')
        if not os.path.exists(self.depth_output_dir):
            os.makedirs(self.depth_output_dir)
        self.normal_output_dir = os.path.join(self.output_dir, 'normal/mat')
        if not os.path.exists(self.normal_output_dir):
            os.makedirs(self.normal_output_dir)
        self.reflectance_output_dir = os.path.join(self.output_dir, 'reflectance/mat')
        if not os.path.exists(self.reflectance_output_dir):
            os.makedirs(self.reflectance_output_dir)
        self.illumination_output_dir = os.path.join(self.output_dir, 'illumination/mat')
        if not os.path.exists(self.illumination_output_dir):
            os.makedirs(self.illumination_output_dir)

        self.model.eval()
        scale = [0.5, 1, 1.5]
        tbar = tqdm(self.test_loader, desc='\r')
        for i, image in enumerate(tbar):
            name = self.test_loader.dataset.images_name[i]
            image = image[0]
            image_in = image.numpy().transpose((1, 2, 0))
            _, H, W = image.shape
            multi_fuse_depth = np.zeros((H, W), np.float32)
            multi_fuse_normal = np.zeros((H, W), np.float32)
            multi_fuse_reflectance = np.zeros((H, W), np.float32)
            multi_fuse_illumination = np.zeros((H, W), np.float32)

            for k in range(0, len(scale)):
                im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                im_ = im_.transpose((2, 0, 1))

                with torch.no_grad():
                    results = self.model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))

                result = torch.squeeze(results[-1].detach()).cpu().numpy()
                res_depth = result[0, :, :]
                res_normal = result[1, :, :]
                res_reflectance = result[2, :, :]
                res_illumination = result[3, :, :]

                fuse_depth = cv2.resize(res_depth, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_fuse_depth += fuse_depth
                fuse_normal = cv2.resize(res_normal, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_fuse_normal += fuse_normal
                fuse_reflectance = cv2.resize(res_reflectance, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_fuse_reflectance += fuse_reflectance
                fuse_illumination = cv2.resize(res_illumination, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_fuse_illumination += fuse_illumination

            multi_fuse_depth = multi_fuse_depth / len(scale)
            multi_fuse_normal = multi_fuse_normal / len(scale)
            multi_fuse_reflectance = multi_fuse_reflectance / len(scale)
            multi_fuse_illumination = multi_fuse_illumination / len(scale)

            sio.savemat(os.path.join(self.depth_output_dir, '{}.mat'.format(name)), {'result': multi_fuse_depth})
            sio.savemat(os.path.join(self.normal_output_dir, '{}.mat'.format(name)), {'result': multi_fuse_normal})
            sio.savemat(os.path.join(self.reflectance_output_dir, '{}.mat'.format(name)), {'result': multi_fuse_reflectance})
            sio.savemat(os.path.join(self.illumination_output_dir, '{}.mat'.format(name)), {'result': multi_fuse_illumination})


def main():
    options = RCF_Options()
    args = options.parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    args.checkname = 'rcf'
    args.lr = 1e-7
    args.minlr = 1e-7
    args.total_iters = 80000
    args.start_iters = 0
    args.display = 20
    print(args)

    trainer = Trainer(args)
    print('Starting iters:', trainer.args.start_iters)
    print('Total iters:', trainer.args.total_iters)
    trainer.training()

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()

if __name__ == "__main__":
    main()
