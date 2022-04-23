import os
import sys
import time
import glob
from pathlib import Path
import numpy as np
import torch
import logging
import argparse
from argparse import ArgumentParser
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from darts.model.model import NetworkCIFAR as Network
from darts.dataset.cifar10 import prepare_cifar10_train
import darts.utils as utils
from darts.bin.run_test import inference
import darts.genotype as genotypes

CIFAR_CLASSES = 10

def cli() -> ArgumentParser:
    parser = argparse.ArgumentParser("cifar")

    parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=20, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument("--auxiliary_weight", type=float, default=0.4, help="weight for auxiliary loss")
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
    parser.add_argument('--save', type=str, default="TRAIN", help="experiment name")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--arch", type=str, default="DARTS", help='which architecture to use')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

    return parser

def train(
    loader_train: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: nn.Module,
    args: argparse.Namespace,
):
    terr = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for idx, (input, label) in enumerate(loader_train):
        input = input.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, label)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, label)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.compute_accuracy(logits, label, topk=(1, 5))
        N = input.shape[0]
        terr.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if idx % args.report_freq == 0:
            logging.info('train %03d %e %f %f', idx, terr.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg

def main():
    args = cli().parse_args()

    # basic logging configurations
    args.save = Path("experiments") / "search-{}-{}".format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt="%m/%d %I:%M:%S %p")
    fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    cudnn.benchmark = True
    cudnn.enabled=True
    torch.cuda.set_device(args.gpu)

    # initialization
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logging.info("gpu device = %d" % args.gpu)
    logging.info("args = %s", args)

    # model
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype, args.drop_path_prob)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
        args.momentum, args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # dataset
    dataloader_train, dataloader_test = prepare_cifar10_train(args)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info("epoch %d lr %e", epoch, lr)

        model._drop_path_prob = args.drop_path_prob * epoch / args.epochs

        # train
        train_acc, _ = train(dataloader_train, model, criterion, optimizer, args)
        logging.info("train_acc %f", train_acc)

        # validation
        test_acc, _ = inference(dataloader_test, model, criterion, args)
        logging.info("test_acc %f", test_acc)

        utils.save(model, os.path.join(args.save, "weights.pt"))


if __name__ == "__main__":
    main()







