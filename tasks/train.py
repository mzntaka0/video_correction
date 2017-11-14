# -*- coding: utf-8 -*-
"""
"""
import os
import sys
import time
import argparse
sys.path.append("./")

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm, trange
from modules.errors import FileNotFoundError, GPUNotFoundError, UnknownOptimizationMethodError
from modules.dataset_indexing.pytorch import PoseDataset, Crop, RandomNoise, Scale
from bpdb import set_trace


def main():
    """ Main function. """
    # arg definition
    parser = argparse.ArgumentParser(
        description='Training pose net for comparison \
        between chainer and pytorch about implementing DeepPose.')
    parser.add_argument(
        'mode', type=str, choices=['chainer', 'pytorch'], help='Mode of training pose net.')
    parser.add_argument(
        '--Nj', '-j', type=int, default=14, help='Number of joints.')
    parser.add_argument(
        '--use-visibility', '-v', action='store_true', help='Use visibility to compute loss.')
    parser.add_argument(
        '--data-augmentation', '-a', action='store_true', help='Crop randomly and add random noise for data augmentation.')
    parser.add_argument(
        '--epoch', '-e', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument(
        '--opt', '-o', type=str, default='MomentumSGD',
        choices=['MomentumSGD', 'Adam'], help='Optimization method.')
    parser.add_argument(
        '--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU).')
    parser.add_argument(
        '--seed', '-s', type=int, help='Random seed to train.')
    parser.add_argument(
        '--train', type=str, default='data/train', help='Path to training image-pose list file.')
    parser.add_argument(
        '--val', type=str, default='data/test', help='Path to validation image-pose list file.')
    parser.add_argument(
        '--batchsize', type=int, default=32, help='Learning minibatch size.')
    parser.add_argument(
        '--out', default='result', help='Output directory')
    parser.add_argument(
        '--resume', default=None,
        help='Initialize the trainer from given file. \
        The file name is "epoch-{epoch number}.iter".')
    parser.add_argument(
        '--resume-model', type=str, default=None,
        help='Load model definition file to use for resuming training \
        (it\'s necessary when you resume a training). \
        The file name is "epoch-{epoch number}.model"')
    parser.add_argument(
        '--resume-opt', type=str, default=None,
        help='Load optimization states from this file \
        (it\'s necessary when you resume a training). \
        The file name is "epoch-{epoch number}.state"')
    args = parser.parse_args()
    args_dict = vars(args)
    trainer = TrainNet
    train = trainer(**args_dict)
    train.start()


class TrainLogger(object):
    """ Logger of training pose net.
    Args:
        out (str): Output directory.
    """

    def __init__(self, out):
        try:
            os.makedirs(out)
        except OSError:
            pass
        self.file = open(os.path.join(out, 'log'), 'w')
        self.logs = []

    def write(self, log):
        """ Write log. """
        tqdm.write(log)
        tqdm.write(log, file=self.file)
        self.logs.append(log)

    def state_dict(self):
        """ Returns the state of the logger. """
        return {'logs': self.logs}

    def load_state_dict(self, state_dict):
        """ Loads the logger state. """
        self.logs = state_dict['logs']
        # write logs.
        tqdm.write(self.logs[-1])
        for log in self.logs:
            tqdm.write(log, file=self.file)


class TrainNet(object):
    """ Train pose net of estimating 2D pose from image.
    Args:
        Nj (int): Number of joints.
        use_visibility (bool): Use visibility to compute loss.
        epoch (int): Number of epochs to train.
        opt (str): Optimization method.
        gpu (bool): Use GPU.
        train (str): Path to training image-pose list file.
        val (str): Path to validation image-pose list file.
        batchsize (int): Learning minibatch size.
        out (str): Output directory.
        resume (str): Initialize the trainer from given file.
            The file name is 'epoch-{epoch number}.iter'.
        resume_model (str): Load model definition file to use for resuming training
            (it\'s necessary when you resume a training).
            The file name is 'epoch-{epoch number}.model'.
        resume_opt (str): Load optimization states from this file
            (it\'s necessary when you resume a training).
            The file name is 'epoch-{epoch number}.state'.
    """

    def __init__(self, **kwargs):
        self.Nj = kwargs['Nj']  # 1
        self.use_visibility = kwargs['use_visibility']  # False
        self.data_augmentation = kwargs['data_augmentation']
        self.epoch = kwargs['epoch']  # 100
        self.gpu = (kwargs['gpu'] >= 0)  # True
        self.opt = kwargs['opt']  # 'Adam'
        self.train = kwargs['train']
        self.val = kwargs['val']
        self.batchsize = kwargs['batchsize']  # 100 
        self.out = kwargs['out']  # 'storage/outputs'
        self.resume = kwargs['resume']  # 'None'
        self.resume_model = kwargs['resume_model']  # None
        self.resume_opt = kwargs['resume_opt']  # None
        # validate arguments.
        self._validate_arguments()

    def _validate_arguments(self):
        if self.gpu and not torch.cuda.is_available():
            raise GPUNotFoundError('GPU is not found.')
        for path in (self.train, self.val):
            if not os.path.isfile(path):
                raise FileNotFoundError('{0} is not found.'.format(path))
        if self.opt not in ('MomentumSGD', 'Adam'):
            raise UnknownOptimizationMethodError(
                '{0} is unknown optimization method.'.format(self.opt))
        if self.resume is not None:
            for path in (self.resume, self.resume_model, self.resume_opt):
                if not os.path.isfile(path):
                    raise FileNotFoundError('{0} is not found.'.format(path))

    def _get_optimizer(self, model):
        if self.opt == 'MomentumSGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        elif self.opt == "Adam":
            optimizer = torch.optim.Adam(model.parameters())
        return optimizer

    def _train(self, model, optimizer, train_iter, log_interval, logger, start_time):
        model.train()
        for iteration, batch in enumerate(tqdm(train_iter, desc='this epoch')):
            image, target = Variable(batch[0]), Variable(batch[1])
            if self.gpu:
                image, target = image.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(image)
            print(output)
            loss = mean_squared_error(output, target, self.use_visibility)
            loss.backward()
            optimizer.step()
            if iteration % log_interval == 0:
                log = 'elapsed_time: {0}, loss: {1}'.format(time.time() - start_time, loss.data[0])
                logger.write(log)

    def _test(self, model, test_iter, logger, start_time):
        model.eval()
        test_loss = 0
        for batch in test_iter:
            image, target  = Variable(batch[0]), Variable(batch[1])
            if self.gpu:
                image, target  = image.cuda(), target.cuda()
            output = model(image)
            test_loss += mean_squared_error(output, target, self.use_visibility).data[0]
        test_loss /= len(test_iter)
        log = 'elapsed_time: {0}, validation/loss: {1}'.format(time.time() - start_time, test_loss)
        logger.write(log)

    def _checkpoint(self, epoch, model, optimizer, logger):
        filename = os.path.join(self.out, 'pytorch', 'epoch-{0}'.format(epoch))
        torch.save({'epoch': epoch + 1, 'logger': logger.state_dict()}, filename + '.iter')
        torch.save(model.state_dict(), filename + '.model')
        torch.save(optimizer.state_dict(), filename + '.state')

    def start(self):
        """ Train pose net. """
        # initialize model to train.
        model = AlexNet(self.Nj)
        if self.resume_model:
            model.load_state_dict(torch.load(self.resume_model))
        # prepare gpu.
        if self.gpu:
            model.cuda()
        input_transforms = [transforms.ToTensor()]
        if self.data_augmentation:
            input_transforms.append(RandomNoise())
        # load the datasets.
        train = PoseDataset(
            self.train,
            input_transform=transforms.Compose(input_transforms),
            output_transform=Scale(),
            transform=Crop(data_augmentation=self.data_augmentation))
        val = PoseDataset(
            self.val,
            input_transform=transforms.Compose([
                transforms.ToTensor()]),
            output_transform=Scale(),
            transform=Crop(data_augmentation=False))
        # training/validation iterators.
        train_iter = torch.utils.data.DataLoader(train, batch_size=self.batchsize, shuffle=True)
        val_iter = torch.utils.data.DataLoader(val, batch_size=self.batchsize, shuffle=False)
        # set up an optimizer.
        optimizer = self._get_optimizer(model)
        if self.resume_opt:
            optimizer.load_state_dict(torch.load(self.resume_opt))
        # set intervals.
        val_interval = 10
        resume_interval = self.epoch/10
        log_interval = 10
        # set logger and start epoch.
        logger = TrainLogger(os.path.join(self.out, 'pytorch'))
        start_epoch = 0
        if self.resume:
            resume = torch.load(self.resume)
            start_epoch = resume['epoch']
            logger.load_state_dict(resume['logger'])
        # start training.
        start_time = time.time()
        for epoch in trange(start_epoch, self.epoch + 1, desc='     total'):
            self._train(model, optimizer, train_iter, log_interval, logger, start_time)
            if epoch % val_interval == 0:
                self._test(model, val_iter, logger, start_time)
            if epoch % resume_interval == 0:
                self._checkpoint(epoch, model, optimizer, logger)

class AlexNet(nn.Module):

    def __init__(self, Nj=1):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        self.fc6 = nn.Linear(256*6*6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, Nj*2)
        self.Nj = Nj

    def forward(self, x):
        # layer1
        h = F.relu(self.conv1(x))
        h = F.max_pool2d(h, 3, stride=2)
        # layer2
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 3, stride=2)
        # layer3-5
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.max_pool2d(h, 3, stride=2)
        h = h.view(-1, 256*6*6)
        # layer6-8
        h = F.dropout(F.relu(self.fc6(h)), training=self.training)
        h = F.dropout(F.relu(self.fc7(h)), training=self.training)
        h = self.fc8(h)
        return h.view(-1, self.Nj, 2)



class MeanSquaredError(nn.Module):
    """ Mean squared error (a.k.a. Euclidean loss) function. """

    def __init__(self, use_visibility=False):
        super(MeanSquaredError, self).__init__()
        self.use_visibility = use_visibility

    def forward(self, *inputs):
        x, t = inputs
        diff = x - t
        if self.use_visibility:
            N = (v.sum()/2).data[0]
            diff = diff*v
        else:
            N = diff.numel()/2
        diff = diff.view(-1)
        return diff.dot(diff)/N

def mean_squared_error(x, t, use_visibility=False):
    """ Computes mean squared error over the minibatch.
    Args:
        x (Variable): Variable holding an float32 vector of estimated pose.
        t (Variable): Variable holding an float32 vector of ground truth pose.
        v (Variable): Variable holding an int32 vector of ground truth pose's visibility.
            (0: invisible, 1: visible)
        use_visibility (bool): When it is ``True``,
            the function uses visibility to compute mean squared error.
    Returns:
        Variable: A variable holding a scalar of the mean squared error loss.
    """
    return MeanSquaredError(use_visibility)(x, t)


if __name__ == '__main__':
    main()
