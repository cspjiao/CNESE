# coding=utf-8
import argparse
import os
import warnings

from models.model import CNESE
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.data import DataLoader
from utils import *

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        "Deep Recursive Network Embedding with Regular Equivalence")
    parser.add_argument('--dataset', type=str, default="barbell",
                        help='Directory to load data.')
    parser.add_argument('-s', '--struct', type=str, default="-1,64,64",
                        help='the network struct')
    parser.add_argument('-e', '--epoch', type=int, default=50,
                        help='Number of epoch to train. Each epoch processes the training '
                             'data once completely')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='Number of training examples processed per step')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--num_hist", type=int, default=80,
                        help="num_hist")
    parser.add_argument('-a', '--alpha', type=float, default=1,
                        help='the rate of vae loss')
    parser.add_argument('--beta', type=float, default=1,
                        help='the rate of gan-node loss')
    parser.add_argument('-g', '--gamma', type=float, default=1,
                        help='the rate of gan-relation loss')
    parser.add_argument('--sampling', type=int,
                        default=50, help='sample number')
    parser.add_argument('--num_workers', type=int,
                        default=4, help='num of workers')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--split_ratio', type=float, default=0.7, help='ratio to split the '
                                                                       'train data')
    parser.add_argument('--loop', type=int, default=100,
                        help='num of classification')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='L2')
    parser.add_argument('--ricci', type=float, default=0.5, help='Ricci')
    parser.add_argument('--device', type=int, default=0, help='GPU')
    parser.add_argument('--no-classification', dest='classification', action='store_false',
                        help='classification')
    return parser.parse_args()


def _main(args):
    graph_path = '../dataset/clf/{}.edge'.format(args.dataset)
    lbl_path = '../dataset/clf/{}.lbl'.format(args.dataset)
    save_path = '../embed/CNESE/{}/embedding/'.format(args.dataset)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    G = load_data(graph_path)

    if not os.path.exists('cache'):
        os.makedirs('cache')

    if not os.path.exists('cache/{}.npy'.format(args.dataset)):
        if args.dataset == 'brazil-flights':
            features = test_feature(G, alpha=0.05, method="OTD", verbose="INFO",
                                    num_hist=args.num_hist)
        elif args.dataset == 'europe-flights':
            features = test_feature(G, alpha=0.9, method="OTD", verbose="INFO",
                                    num_hist=args.num_hist)
        else:
            features = test_feature(G, alpha=0.5, method="OTD", verbose="INFO",
                                    num_hist=args.num_hist)
            np.save('cache/{}.npy'.format(args.dataset), features)
    else:
        features = np.load('cache/{}.npy'.format(args.dataset))

    print("Nodes: {}, Edges: {}".format(
        G.number_of_nodes(), G.number_of_edges()))
    print("Features size: {}".format(features.shape))
    device = torch.device(
        "cuda:{}".format(args.device) if args.device >= 0 else "cpu")
    args.device = device
    args.struct = list(map(lambda x: int(x), args.struct.split(',')))

    model = CNESE(args, G, features).to(device, dtype=torch.float32)

    optimizer = torch.optim.Adam([
        {'params': model.vae.parameters(), 'weight_decay': 0},
        {'params': model.mlp.parameters(), },
    ], lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_d = torch.optim.Adam([
        {'params': model.discriminator.parameters()},
    ], lr=args.learning_rate, weight_decay=args.weight_decay)
    for epoch in range(args.epoch):
        train_dataloader = DataLoader(list(G.nodes), args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)
        total_loss = 0.0
        total_loss_g = 0.0
        total_loss_d = 0.0

        for idx, data in enumerate(train_dataloader):
            nodes = data
            for _ in range(5):
                optimizer.zero_grad()
                loss_g, loss_d = model(nodes)
                loss_g.backward()
                optimizer.step()

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()
            total_loss += loss_g + loss_d
            total_loss_g += loss_g
            total_loss_d += loss_d
        print('epoch: {}, Generator loss: {}, Discriminator loss: {}, Total loss: {}'.format(
            epoch, total_loss_g.item(), total_loss_d.item(), total_loss.item()))

        embedding = model.get_embedding()
        embedding = embedding.data.cpu().numpy()
        np.save("{}{}".format(save_path, epoch), embedding)

        if args.classification:
            eval_dict = classification(embedding, lbl_path,
                                       split_ratio=args.split_ratio,
                                       loop=args.loop)

    embedding = np.load("{}{}.npy".format(save_path, args.epoch - 1))
    columns = ["id"] + ["x_" + str(x) for x in range(embedding.shape[1])]
    ids = np.array(list(range(embedding.shape[0]))).reshape((-1, 1))
    embedding = pd.DataFrame(np.concatenate(
        [ids, embedding], axis=1), columns=columns)
    embedding = embedding.sort_values(by=['id'])
    print('Save best embedding to ../embed/CNESE/clf/{}.emb'.format(args.dataset))
    embedding.to_csv(
        "../embed/CNESE/clf/{}.emb".format(args.dataset), index=False)


if __name__ == '__main__':
    _main(parse_args())
