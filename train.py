import paddle
import os.path as osp
import sys
import argparse
from mmcv import Config
import os
import time
from collections import OrderedDict
from dataset import build_data_loader
from models import build_model
from utils import AverageMeter


def train(train_loader, model, optimizer, epoch, start_iter, cfg):
    model.train()

    # meters
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    losses_text = AverageMeter()
    losses_kernels = AverageMeter()

    ious_text = AverageMeter()
    ious_kernel = AverageMeter()

    # start time
    start = time.time()

    for iter, data_ in enumerate(train_loader):

        # skip previous iterations
        if iter < start_iter:
            print('Skipping iter: %d' % iter)
            sys.stdout.flush()
            continue

        # time cost of data loader
        data_time.update(time.time() - start)

        adjust_learning_rate(optimizer, train_loader, epoch, iter, cfg)

        # prepare input
        data = dict(
            imgs=data_[0],
            gt_texts=data_[1],
            gt_kernels=data_[2],
            training_masks=data_[3],
        )
        data.update(dict(cfg=cfg))
        outputs = model(**data)

        # detection loss
        loss_text = paddle.mean(outputs['loss_text'])
        losses_text.update(float(loss_text))

        loss_kernels = paddle.mean(outputs['loss_kernels'])
        losses_kernels.update(float(loss_kernels))

        loss = loss_text + loss_kernels

        iou_text = paddle.mean(outputs['iou_text'])
        ious_text.update(float(iou_text))
        iou_kernel = paddle.mean(outputs['iou_kernel'])
        ious_kernel.update(float(iou_kernel))

        losses.update(float(loss))
        # backward
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)

        # update start time
        start = time.time()

        # print log
        if iter % 20 == 0:
            output_log = '({batch}/{size}) LR: {lr:.6f} | Batch: {bt:.3f}s | Total: {total:.0f}min | ' \
                         'ETA: {eta:.0f}min | Loss: {loss:.3f} | ' \
                         'Loss(text/kernel): {loss_text:.3f}/{loss_kernel:.3f} ' \
                         '| IoU(text/kernel): {iou_text:.3f}/{iou_kernel:.3f} '.format(
                batch=iter + 1,
                size=len(train_loader),
                lr=optimizer.get_lr(),
                bt=batch_time.avg,
                total=batch_time.avg * iter / 60.0,
                eta=batch_time.avg * (len(train_loader) - iter) / 60.0,
                loss=losses.avg,
                loss_text=losses_text.avg,
                loss_kernel=losses_kernels.avg,
                iou_text=ious_text.avg,
                iou_kernel=ious_kernel.avg,
            )
            print(output_log)
            sys.stdout.flush()


def adjust_learning_rate(optimizer, dataloader, epoch, iter, cfg):
    schedule = cfg.train_cfg.schedule
    if isinstance(schedule, str):
        assert schedule == 'polylr', 'Error: schedule should be polylr!'
        cur_iter = epoch * len(dataloader) + iter
        max_iter_num = cfg.train_cfg.epoch * len(dataloader)
        lr = cfg.train_cfg.lr * (1 - float(cur_iter) / max_iter_num) ** 0.9
    elif isinstance(schedule, tuple):
        lr = cfg.train_cfg.lr
        for i in range(len(schedule)):
            if epoch < schedule[i]:
                break
            lr = lr * 0.1

    optimizer.set_lr(lr)


def save_checkpoint(state, checkpoint_path, cfg):
    file_path = osp.join(checkpoint_path, 'checkpoint.pd')
    paddle.save(state, file_path)
    if cfg.data.train.type in ['synth'] or \
            (state['iter'] == 0 and state['epoch'] > cfg.train_cfg.epoch - 100 and state['epoch'] % 10 == 0):
        file_name = 'checkpoint_%dep.pd' % state['epoch']
        file_path = osp.join(checkpoint_path, file_name)
        paddle.save(state, file_path)


def main(args):
    cfg = Config.fromfile(args.config)

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        cfg_name, _ = osp.splitext(osp.basename(args.config))
        checkpoint_path = osp.join('checkpoints', cfg_name)
    if not osp.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    print('Checkpoint path: %s.' % checkpoint_path)

    # data loader
    data_set = build_data_loader(cfg.data.train)
    train_loader = paddle.io.DataLoader(
        data_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        use_shared_memory=True
    )

    model = build_model(cfg.model)

    if cfg.train_cfg.optimizer == 'SGD':
        optimizer = paddle.optimizer.SGD(learning_rate=cfg.train_cfg.lr, parameters=model.parameters(),
                                         weight_decay=5e-4)
    elif cfg.train_cfg.optimizer == 'Adam':
        optimizer = paddle.optimizer.Adam(learning_rate=cfg.train_cfg.lr, parameters=model.parameters())

    start_epoch = 0
    start_iter = 0
    if hasattr(cfg.train_cfg, 'pretrain'):
        assert osp.isfile(cfg.train_cfg.pretrain), 'Error: no pretrained weights found!'
        print('Finetuning from pretrained model %s.' % cfg.train_cfg.pretrain)
        checkpoint = paddle.load(cfg.train_cfg.pretrain)
        model.set_state_dict(checkpoint['state_dict'])
    if args.resume:
        assert osp.isfile(args.resume), 'Error: no checkpoint directory found!'
        print('Resuming from checkpoint %s.' % args.resume)
        checkpoint = paddle.load(args.resume)
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint['iter']
        model.set_state_dict(checkpoint['state_dict'])
        optimizer.set_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch, cfg.train_cfg.epoch):
        print('\nEpoch: [%d | %d]' % (epoch + 1, cfg.train_cfg.epoch))

        train(train_loader, model, optimizer, epoch, start_iter, cfg)

        state = OrderedDict(
            epoch=epoch + 1,
            iter=0,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )
        save_checkpoint(state, checkpoint_path, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--resume', nargs='?', type=str, default=None)
    args = parser.parse_args()
    main(args)
