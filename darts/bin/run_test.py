from operator import mod
import os
import sys
import glob
import numpy as np
import torch
import logging
import argparse
from argparse import ArgumentParser
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from typing import Tuple

import darts.genotype as genotypes
from darts.model.model import NetworkCIFAR as Network
import darts.utils as utils
from darts.dataset.cifar10 import prepare_cifar10_test

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 10

def cli() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="data", help="location of the data corpus")
    parser.add_argument("--batch_size", type=int, default=96, help="batch size")
    parser.add_argument("--report_freq", type=float, default=20, help="report frequency")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument("--init_channels", type=int, default=36, help="num of init channels")
    parser.add_argument("--layers", type=int, default=20, help="total number of layers")
    parser.add_argument("--model_path", type=str, default="weights/cifar10_model.pt", help="path of pretrained model")
    parser.add_argument("--auxiliary", action="store_true", default=False, help="use auxiliary tower")
    parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
    parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
    parser.add_argument("--drop_path_prob", type=float, default=0.2, help="drop path probability")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--arch", type=str, default="DARTS", help="which architecture to use")

    return parser

def inference(dataloader: DataLoader, model: nn.Module, criterion: nn.Module, args: argparse.Namespace) -> Tuple[float, float]:
    terr = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.eval()
    for idx, (input, label) in enumerate(dataloader):
        input = input.requires_grad_().cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # forward
        logits, _ = model(input)
        loss = criterion(logits, label)

        prec1, prec5 = utils.compute_accuracy(logits, label, (1, 5))
        N = input.shape[0]
        terr.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if idx % args.report_freq == 0:
            logging.info("test %03d %e %f %f", idx, terr.avg, top1.avg, top5.avg)

    return top1.avg, terr.avg


def main():
    args = cli().parse_args()

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    cudnn.enabled=True
    cudnn.benchmark = True
    torch.cuda.set_device(args.gpu)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # model
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(C=args.init_channels, num_classes=CIFAR_CLASSES, layers=args.layers,
        auxiliary=args.auxiliary, genotype=genotype, drop_path_prob=args.drop_path_prob)
    model = model.cuda()
    utils.load(model, args.model_path)

    logging.info("param size = %.4fMB", utils.count_parameters_in_MB(model))

    # criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # dataset
    if not os.path.exists(args.data):
        os.mkdir(args.data)
    test_dataloader = prepare_cifar10_test(args)

    test_acc, _ = inference(test_dataloader, model, criterion, args)
    logging.info("\n%s\n %s test_acc %f \n%s", "=" * 65, " " * 20, test_acc, "=" * 65)


if __name__ == "__main__":
    main()



