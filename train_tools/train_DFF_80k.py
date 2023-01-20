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
import modeling.dff_encoding.utils as utils
from utils.DFF_losses import EdgeDetectionReweightedLosses
from modeling.DFF import DFF
from my_options.DFF_options import Options
from modeling.sync_batchnorm.replicate import patch_replication_callback
from utils.saver import Saver
from utils.summaries import TensorboardSummary
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
        self.class_num = 4
        # Define network
        model = DFF(self.class_num, backbone=self.args.backbone)

        # optimizer using different LR
        if args.model == 'dff':  # dff
            params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},
                           {'params': model.ada_learner.parameters(), 'lr': args.lr * 10},
                           {'params': model.side1.parameters(), 'lr': args.lr * 10},
                           {'params': model.side2.parameters(), 'lr': args.lr * 10},
                           {'params': model.side3.parameters(), 'lr': args.lr * 10},
                           {'params': model.side5.parameters(), 'lr': args.lr * 10},
                           {'params': model.side5_w.parameters(), 'lr': args.lr * 10}]
        else:  # casenet
            assert args.model == 'casenet'
            params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},
                           {'params': model.side1.parameters(), 'lr': args.lr * 10},
                           {'params': model.side2.parameters(), 'lr': args.lr * 10},
                           {'params': model.side3.parameters(), 'lr': args.lr * 10},
                           {'params': model.side5.parameters(), 'lr': args.lr * 10},
                           {'params': model.fuse.parameters(), 'lr': args.lr * 10}]

        optimizer = torch.optim.SGD(params_list,
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        self.criterion = EdgeDetectionReweightedLosses()
        self.model, self.optimizer = model, optimizer

        # Define lr scheduler
        self.scheduler = PolyLrUpdaterHook(power=0.9, base_lr=self.args.lr, min_lr=self.args.minlr)
        # Using cuda
        if self.args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

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

            outputs = self.model(image.float())
            loss = self.criterion(outputs, target)

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
        print('Test iters: %d' % iters)
        '''
        self.output_dir = os.path.join(self.saver.experiment_dir, 'side_outs', str(iters + 1))
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
        '''

        self.output_dir2 = os.path.join(self.saver.experiment_dir, 'fuse_outs', str(iters + 1))
        if not os.path.exists(self.output_dir2):
            os.makedirs(self.output_dir2)
        self.depth_output_dir2 = os.path.join(self.output_dir2, 'depth/mat')
        if not os.path.exists(self.depth_output_dir2):
            os.makedirs(self.depth_output_dir2)
        self.normal_output_dir2 = os.path.join(self.output_dir2, 'normal/mat')
        if not os.path.exists(self.normal_output_dir2):
            os.makedirs(self.normal_output_dir2)
        self.reflectance_output_dir2 = os.path.join(self.output_dir2, 'reflectance/mat')
        if not os.path.exists(self.reflectance_output_dir2):
            os.makedirs(self.reflectance_output_dir2)
        self.illumination_output_dir2 = os.path.join(self.output_dir2, 'illumination/mat')
        if not os.path.exists(self.illumination_output_dir2):
            os.makedirs(self.illumination_output_dir2)

        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        for i, image in enumerate(tbar):
            name = self.test_loader.dataset.images_name[i]
            if self.args.cuda:
                image = image.cuda()
                crop_h, crop_w = image.size(2), image.size(3)
                image = image[:, :, 0:crop_h - 1, 0:crop_w - 1]
            with torch.no_grad():
                output_list = self.model(image)

            '''
            pred2 = output_list[0]
            pred2 = torch.sigmoid(pred2)

            pred = torch.zeros(1, 4, crop_h, crop_w)
            pred[:, :, 0:crop_h - 1, 0:crop_w - 1] = pred2

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
            '''

            pred2 = output_list[1]
            pred2 = torch.sigmoid(pred2)

            pred = torch.zeros(1, 4, crop_h, crop_w)
            pred[:, :, 0:crop_h - 1, 0:crop_w - 1] = pred2

            pred = pred.squeeze()
            out_depth = pred[0, :, :]
            out_normal = pred[1, :, :]
            out_reflectance = pred[2, :, :]
            out_illumination = pred[3, :, :]

            depth_pred = out_depth.data.cpu().numpy()
            depth_pred = depth_pred.squeeze()
            sio.savemat(os.path.join(self.depth_output_dir2, '{}.mat'.format(name)), {'result': depth_pred})

            normal_pred = out_normal.data.cpu().numpy()
            normal_pred = normal_pred.squeeze()
            sio.savemat(os.path.join(self.normal_output_dir2, '{}.mat'.format(name)), {'result': normal_pred})

            reflectance_pred = out_reflectance.data.cpu().numpy()
            reflectance_pred = reflectance_pred.squeeze()
            sio.savemat(os.path.join(self.reflectance_output_dir2, '{}.mat'.format(name)), {'result': reflectance_pred})

            illumination_pred = out_illumination.data.cpu().numpy()
            illumination_pred = illumination_pred.squeeze()
            sio.savemat(os.path.join(self.illumination_output_dir2, '{}.mat'.format(name)),
                        {'result': illumination_pred})

    def multiscale_test(self, iters):
        print('Test epoch: %d' % iters)
        '''
        self.output_dir = os.path.join(self.saver.experiment_dir, 'side_outs_ms', str(iters + 1))
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
        '''

        self.output_dir2 = os.path.join(self.saver.experiment_dir, 'fuse_outs_ms', str(iters + 1))
        if not os.path.exists(self.output_dir2):
            os.makedirs(self.output_dir2)
        self.depth_output_dir2 = os.path.join(self.output_dir2, 'depth/mat')
        if not os.path.exists(self.depth_output_dir2):
            os.makedirs(self.depth_output_dir2)
        self.normal_output_dir2 = os.path.join(self.output_dir2, 'normal/mat')
        if not os.path.exists(self.normal_output_dir2):
            os.makedirs(self.normal_output_dir2)
        self.reflectance_output_dir2 = os.path.join(self.output_dir2, 'reflectance/mat')
        if not os.path.exists(self.reflectance_output_dir2):
            os.makedirs(self.reflectance_output_dir2)
        self.illumination_output_dir2 = os.path.join(self.output_dir2, 'illumination/mat')
        if not os.path.exists(self.illumination_output_dir2):
            os.makedirs(self.illumination_output_dir2)

        self.model.eval()
        scale = [0.5, 1, 1.5]
        tbar = tqdm(self.test_loader, desc='\r')
        for i, image in enumerate(tbar):
            name = self.test_loader.dataset.images_name[i]
            image = image[0]
            crop_h, crop_w = image.size(1), image.size(2)
            image = image[:, 0:crop_h - 1, 0:crop_w - 1]

            image_in = image.numpy().transpose((1, 2, 0))
            _, H, W = image.shape
            '''
            multi_side_depth = np.zeros((H, W), np.float32)
            multi_side_normal = np.zeros((H, W), np.float32)
            multi_side_reflectance = np.zeros((H, W), np.float32)
            multi_side_illumination = np.zeros((H, W), np.float32)
            '''
            multi_fuse_depth = np.zeros((H, W), np.float32)
            multi_fuse_normal = np.zeros((H, W), np.float32)
            multi_fuse_reflectance = np.zeros((H, W), np.float32)
            multi_fuse_illumination = np.zeros((H, W), np.float32)

            for k in range(0, len(scale)):
                im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                im_ = im_.transpose((2, 0, 1))

                with torch.no_grad():
                    pred_side, pred_fuse = self.model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
                    pred_side = torch.sigmoid(pred_side)
                    pred_fuse = torch.sigmoid(pred_fuse)

                '''
                pred_side = torch.squeeze(pred_side.detach()).cpu().numpy()
                depth_side = pred_side[0, :, :]
                normal_side = pred_side[1, :, :]
                reflectance_side = pred_side[2, :, :]
                illumination_side = pred_side[3, :, :]
                depth_side = cv2.resize(depth_side, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_side_depth += depth_side
                normal_side = cv2.resize(normal_side, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_side_normal += normal_side
                reflectance_side = cv2.resize(reflectance_side, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_side_reflectance += reflectance_side
                illumination_side = cv2.resize(illumination_side, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_side_illumination += illumination_side
                '''

                pred_fuse = torch.squeeze(pred_fuse.detach()).cpu().numpy()
                depth_fuse = pred_fuse[0, :, :]
                normal_fuse = pred_fuse[1, :, :]
                reflectance_fuse = pred_fuse[2, :, :]
                illumination_fuse = pred_fuse[3, :, :]
                depth_fuse = cv2.resize(depth_fuse, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_fuse_depth += depth_fuse
                normal_fuse = cv2.resize(normal_fuse, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_fuse_normal += normal_fuse
                reflectance_fuse = cv2.resize(reflectance_fuse, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_fuse_reflectance += reflectance_fuse
                illumination_fuse = cv2.resize(illumination_fuse, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_fuse_illumination += illumination_fuse


            multi_fuse_depth = multi_fuse_depth / len(scale)
            multi_fuse_normal = multi_fuse_normal / len(scale)
            multi_fuse_reflectance = multi_fuse_reflectance / len(scale)
            multi_fuse_illumination = multi_fuse_illumination / len(scale)

            multi_fuse_depth2 = np.zeros((crop_h, crop_w), np.float32)
            multi_fuse_depth2[0:crop_h - 1, 0:crop_w - 1] = multi_fuse_depth
            multi_fuse_normal2 = np.zeros((crop_h, crop_w), np.float32)
            multi_fuse_normal2[0:crop_h - 1, 0:crop_w - 1] = multi_fuse_normal
            multi_fuse_reflectance2 = np.zeros((crop_h, crop_w), np.float32)
            multi_fuse_reflectance2[0:crop_h - 1, 0:crop_w - 1] = multi_fuse_reflectance
            multi_fuse_illumination2 = np.zeros((crop_h, crop_w), np.float32)
            multi_fuse_illumination2[0:crop_h - 1, 0:crop_w - 1] = multi_fuse_illumination

            sio.savemat(os.path.join(self.depth_output_dir2, '{}.mat'.format(name)), {'result': multi_fuse_depth2})
            sio.savemat(os.path.join(self.normal_output_dir2, '{}.mat'.format(name)), {'result': multi_fuse_normal2})
            sio.savemat(os.path.join(self.reflectance_output_dir2, '{}.mat'.format(name)),
                        {'result': multi_fuse_reflectance2})
            sio.savemat(os.path.join(self.illumination_output_dir2, '{}.mat'.format(name)),
                        {'result': multi_fuse_illumination2})

            '''
            multi_side_depth = multi_side_depth / len(scale)
            multi_side_normal = multi_side_normal / len(scale)
            multi_side_reflectance = multi_side_reflectance / len(scale)
            multi_side_illumination = multi_side_illumination / len(scale)

            multi_side_depth2 = np.zeros((crop_h, crop_w), np.float32)
            multi_side_depth2[0:crop_h - 1, 0:crop_w - 1] = multi_side_depth
            multi_side_normal2 = np.zeros((crop_h, crop_w), np.float32)
            multi_side_normal2[0:crop_h - 1, 0:crop_w - 1] = multi_side_normal
            multi_side_reflectance2 = np.zeros((crop_h, crop_w), np.float32)
            multi_side_reflectance2[0:crop_h - 1, 0:crop_w - 1] = multi_side_reflectance
            multi_side_illumination2 = np.zeros((crop_h, crop_w), np.float32)
            multi_side_illumination2[0:crop_h - 1, 0:crop_w - 1] = multi_side_illumination

            sio.savemat(os.path.join(self.depth_output_dir, '{}.mat'.format(name)), {'result': multi_side_depth2})
            sio.savemat(os.path.join(self.normal_output_dir, '{}.mat'.format(name)), {'result': multi_side_normal2})
            sio.savemat(os.path.join(self.reflectance_output_dir, '{}.mat'.format(name)),
                        {'result': multi_side_reflectance2})
            sio.savemat(os.path.join(self.illumination_output_dir, '{}.mat'.format(name)),
                        {'result': multi_side_illumination2})
            '''




def main():
    args = Options().parse()
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

    args.checkname = 'dff'
    args.lr = 1e-8
    args.minlr = 1e-10
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
