import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
from argparse import ArgumentParser
import torch.nn as nn
import torch.utils
from pathlib import Path
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from typing import Tuple

from darts.model.search import Network
import darts.utils as utils
from darts.dataset.cifar10 import transform_cifar10
from darts.dataset.cifar10 import prepare_cifar10_search
from darts.architect import Architect

CIFAR_CLASSES = 10

def cli() -> ArgumentParser:
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=20, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, default="EXP", help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

    return parser

def train(
    loader_train: DataLoader,
    loader_val: DataLoader,
    model: nn.Module,
    architect: object,
    criterion: nn.Module,
    optimizer: nn.Module,
    lr: float,
    args: argparse.Namespace,
) -> Tuple[float, float]:
    terr = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for idx, (input, label) in enumerate(loader_train):
        model.train()
        N = input.shape[0]

        input.requires_grad = False
        input = input.cuda()
        label.requires_grad = False
        label = label.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, label_search = next(iter(loader_val))
        input_search.requires_grad = False
        input_search = input_search.cuda()
        label_search.requires_grad = False
        label_search = label_search.cuda(non_blocking=True)

        architect.step(input, label, input_search, label_search, lr, optimizer, args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, label)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.compute_accuracy(logits, label, topk=(1, 5))
        terr.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if idx % args.report_freq == 0:
            logging.info('train %03d %e %f %f', idx, terr.avg, top1.avg, top5.avg)

    return top1.avg, terr.avg

def inference(
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    args: argparse.Namespace,
) -> Tuple[float, float]:
    terr = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.eval()
    for idx, (input, label) in enumerate(dataloader):
        input = input.cuda()
        label = label.cuda(non_blocking=True)

        logits = model(input)
        loss = criterion(logits, label)

        prec1, prec5 = utils.compute_accuracy(logits, label, topk=(1, 5))
        N = input.shape[0]
        terr.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if idx % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', idx, terr.avg, top1.avg, top5.avg)

    return top1.avg, terr.avg

def main():
    args = cli().parse_args()

    # basic logging configurations
    args.save = Path("experiments") / "search-{}-{}".format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    cudnn.benchmark = True
    cudnn.enabled=True
    torch.cuda.set_device(args.gpu)

    # initialization
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # model
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
        momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        float(args.epochs), eta_min=args.learning_rate_min)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # dataset
    dataloader_train, dataloader_val = prepare_cifar10_search(args)

    # architect
    architect = Architect(model, args)

    # start searching
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info("epoch %d lr %e", epoch, lr)

        genotype = model.genotype()
        logging.info("genotype = %s", genotype)

        logging.info("normal and reduced alphas")
        logging.info(F.softmax(model.alphas_normal, dim=-1))
        logging.info(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, _ = train(dataloader_train, dataloader_val, model, architect, criterion, optimizer, lr, args)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, _ = inference(dataloader_val, model, criterion, args)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, "weights.pt"))


if __name__ == "__main__":
    main()


