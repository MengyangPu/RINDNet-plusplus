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
from dataloaders.datasets.bsds_hd5_dim1 import Mydataset
from torch.utils.data import DataLoader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from my_options.BDCN_options import BDCN_Options
from modeling.BDCN_edge import BDCN
from utils.bdcn_loss import bdcn_loss_edge
from utils.saver import Saver
from utils.summaries import TensorboardSummary
import scipy.io as sio
import time
import re
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
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
        self.model = BDCN(self.args.pretrain_model)
        self.model.cuda()

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
        params_dict = dict(self.model.named_parameters())
        base_lr = args.lr
        weight_decay = args.weight_decay
        params = []
        for key, v in params_dict.items():
            if re.match(r'conv[1-5]_[1-3]_down', key):
                if 'weight' in key:
                    params += [{'params': v, 'lr': base_lr * 0.1, 'weight_decay': weight_decay * 1, 'name': key}]
                elif 'bias' in key:
                    params += [{'params': v, 'lr': base_lr * 0.2, 'weight_decay': weight_decay * 0, 'name': key}]
            elif re.match(r'.*conv[1-4]_[1-3]', key):
                if 'weight' in key:
                    params += [{'params': v, 'lr': base_lr * 1, 'weight_decay': weight_decay * 1, 'name': key}]
                elif 'bias' in key:
                    params += [{'params': v, 'lr': base_lr * 2, 'weight_decay': weight_decay * 0, 'name': key}]
            elif re.match(r'.*conv5_[1-3]', key):
                if 'weight' in key:
                    params += [{'params': v, 'lr': base_lr * 100, 'weight_decay': weight_decay * 1, 'name': key}]
                elif 'bias' in key:
                    params += [{'params': v, 'lr': base_lr * 200, 'weight_decay': weight_decay * 0, 'name': key}]
            elif re.match(r'score_dsn[1-5]', key):
                if 'weight' in key:
                    params += [{'params': v, 'lr': base_lr * 0.01, 'weight_decay': weight_decay * 1, 'name': key}]
                elif 'bias' in key:
                    params += [{'params': v, 'lr': base_lr * 0.02, 'weight_decay': weight_decay * 0, 'name': key}]
            elif re.match(r'upsample_[248](_5)?', key):
                if 'weight' in key:
                    params += [{'params': v, 'lr': base_lr * 0, 'weight_decay': weight_decay * 0, 'name': key}]
                elif 'bias' in key:
                    params += [{'params': v, 'lr': base_lr * 0, 'weight_decay': weight_decay * 0, 'name': key}]
            elif re.match(r'.*msblock[1-5]_[1-3]\.conv', key):
                if 'weight' in key:
                    params += [{'params': v, 'lr': base_lr * 1, 'weight_decay': weight_decay * 1, 'name': key}]
                elif 'bias' in key:
                    params += [{'params': v, 'lr': base_lr * 2, 'weight_decay': weight_decay * 0, 'name': key}]
            else:
                if 'weight' in key:
                    params += [{'params': v, 'lr': base_lr * 0.001, 'weight_decay': weight_decay * 1, 'name': key}]
                elif 'bias' in key:
                    params += [{'params': v, 'lr': base_lr * 0.002, 'weight_decay': weight_decay * 0, 'name': key}]
        self.optimizer = torch.optim.SGD(params, momentum=self.args.momentum,
                                    lr=self.args.lr, weight_decay=self.args.weight_decay)

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
            self.best_pred = checkpoint['best_pred']
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
        batch_size = self.args.batch_size
        self.model.train()
        start_time = time.time()
        for step in range(self.args.start_iters, self.args.total_iters):

            if cur == iter_per_epoch:
                cur = 0
                data_iter = iter(self.train_loader)
            image, target = next(data_iter)
            if self.args.cuda:
                image, target = image.cuda(), target.cuda() #(b,3,w,h) (b,1,w,h)
                target = target.unsqueeze(1)

            out = self.model(image)
            loss = 0
            for k in range(10):
                loss += self.args.side_weight * bdcn_loss_edge(out[k], target, self.args.cuda,
                                                               self.args.balance) / batch_size
            loss += self.args.fuse_weight * bdcn_loss_edge(out[-1], target, self.args.cuda,
                                                           self.args.balance) / batch_size

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
        self.output_dir = os.path.join(self.saver.experiment_dir, str(iters+1), 'mat')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        for i, image in enumerate(tbar):
            name = self.test_loader.dataset.images_name[i]
            if self.args.cuda:
                image = image.cuda()
            with torch.no_grad():
                preds_list = self.model(image)

            pred = preds_list[-1]
            pred = pred.squeeze()
            pred = pred.data.cpu().numpy()
            sio.savemat(os.path.join(self.output_dir, '{}.mat'.format(name)), {'result': pred})


    def multiscale_test(self, iters):
        print('Test epoch: %d' % iters)
        self.output_dir = os.path.join(self.saver.experiment_dir, str(iters + 1) + '_ms', 'mat')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.model.eval()
        scale = [0.5, 1, 1.5]
        tbar = tqdm(self.test_loader, desc='\r')
        for i, image in enumerate(tbar):
            name = self.test_loader.dataset.images_name[i]
            image = image[0]
            image_in = image.numpy().transpose((1, 2, 0))
            _, H, W = image.shape
            multi_fuse = np.zeros((H, W), np.float32)

            for k in range(0, len(scale)):
                im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                im_ = im_.transpose((2, 0, 1))

                with torch.no_grad():
                    results = self.model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))

                result = torch.squeeze(results[-1].detach()).cpu().numpy()

                fuse_depth = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_fuse += fuse_depth

            multi_fuse = multi_fuse / len(scale)

            sio.savemat(os.path.join(self.output_dir, '{}.mat'.format(name)), {'result': multi_fuse})


def main():
    options = BDCN_Options()
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

    args.data_path = 'data/BSDS-RIND/BSDS-RIND-Edge/Augmentation/'
    args.lr = 1e-6
    args.minlr = 1e-8
    args.total_iters = 80000
    args.start_iters = 0
    args.display = 20
    print(args)

    trainer = Trainer(args)
    print('Starting iters:', trainer.args.start_iters)
    print('Total iters:', trainer.args.total_iters)
    trainer.training()


if __name__ == "__main__":
    main()
