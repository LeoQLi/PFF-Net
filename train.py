import os, sys
import time
import argparse
import random
import numpy as np

import torch
import torch.utils.data
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.optim.lr_scheduler as lr_scheduler

# torch.autograd.set_detect_anomaly(True)   # slower! to show more details about errors

from misc import seed_all, prepare
from net.network import Network
from dataset import PointCloudDataset, PatchDataset, RandomPointcloudPatchSampler



def parse_arguments():
    parser = argparse.ArgumentParser()
    ## Training
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--log_root', type=str, default='./log')
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--nepoch', type=int, default=1000)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--max_grad_norm', type=float, default=float("inf"))
    ## Dataset and loader
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--data_set', type=str, default='PCPNet')
    parser.add_argument('--trainset_list', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=0)
    parser.add_argument('--sample_size', type=int, default=0)
    parser.add_argument('--encode_knn', type=int, default=16)
    parser.add_argument('--patches_per_shape', type=int, default=1000,
                        help='The number of patches sampled from each shape in an epoch')
    args = parser.parse_args()
    return args


def get_data_loaders(args):
    def worker_init_fn(worker_id):
        random.seed(args.seed)
        np.random.seed(args.seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dset = PointCloudDataset(
            root=args.dataset_root,
            mode='train',
            data_set=args.data_set,
            data_list=args.trainset_list,
        )
    train_set = PatchDataset(
            datasets=train_dset,
            patch_size=args.patch_size,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    train_datasampler = RandomPointcloudPatchSampler(train_set, patches_per_shape=args.patches_per_shape, seed=args.seed)
    train_dataloader = torch.utils.data.DataLoader(
            train_set,
            sampler=train_datasampler,
            batch_size=args.batch_size,
            num_workers=int(args.num_workers),
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=g,
        )

    return train_dataloader, train_datasampler


def train(epoch):
    for batch_idx, batch in enumerate(train_dataloader, 0):
        pcl_pat = batch['pcl_pat'].to(_device)
        normal_pat = batch['normal_pat'].to(_device)
        normal_center = batch['normal_center'].to(_device).squeeze()                                # (B, 3)

        ### Reset grad and model state
        model.train()
        optimizer.zero_grad()

        ### Forward
        pred_point, weights, pred_neighbor = model(pcl_pat)
        loss, loss_tuple = model.get_loss(q_target=normal_center, q_pred=pred_point,
                                            ne_target=normal_pat, ne_pred=pred_neighbor,
                                            pred_weights=weights, pcl_in=pcl_pat,
                                        )

        ### Backward and optimize
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        ### Logging
        if batch_idx % 100 == 0:
            s = ''
            for l in loss_tuple:
                s += '%.5f+' % l.item()
            logger.info('[Train] [%03d: %03d/%03d] | Loss: %.6f(%s) | Grad: %.6f' % (
                        epoch, batch_idx, train_num_batch-1, loss.item(), s[:-1], orig_grad_norm)
                    )


if __name__ == '__main__':
    ### Arguments
    args = parse_arguments()
    seed_all(args.seed)

    assert args.gpu >= 0, "ERROR GPU ID!"
    _device = torch.device('cuda:%d' % args.gpu)

    ### Model
    print('Building model ...')
    model = Network(encode_knn=args.encode_knn).to(_device)

    ### Datasets and loaders
    print('Loading datasets ...')
    train_dataloader, train_datasampler = get_data_loaders(args)
    train_num_batch = len(train_dataloader)

    ### Optimizer and Scheduler
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.nepoch*0.4), int(args.nepoch*0.6)], gamma=0.2)

    #### Logging
    logger, ckpt_dir = prepare(args)

    ### Arguments
    logger.info('Command: {}'.format(' '.join(sys.argv)))
    arg_str = '\n'.join(['    {}: {}'.format(op, getattr(args, op)) for op in vars(args)])
    logger.info('Arguments:\n' + arg_str)
    logger.info(repr(model))
    logger.info('Training set: %d patches (in %d batches)' % (len(train_datasampler), len(train_dataloader)))
    logger.info('Start training ...')

    try:
        for epoch in range(1, args.nepoch+1):
            logger.info('### Epoch %d ###' % epoch)

            start_time = time.time()
            train(epoch)
            end_time = time.time()
            logger.info('Time cost: %.1f s \n' % (end_time-start_time))

            scheduler.step()

            if epoch % args.interval == 0 or epoch == args.nepoch:
                model_filename = os.path.join(ckpt_dir, 'ckpt_%d.pt' % epoch)
                torch.save(model.state_dict(), model_filename)

    except KeyboardInterrupt:
        logger.info('Terminating ...')
