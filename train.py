import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser
import glob
from pathlib import Path
import re
import json
from detect import detect

import torch
import cv2
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from dataset import CosineAnnealingWarmUpRestarts
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

import wandb
from deteval import calc_deteval_metrics


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/ICDAR17_Korean'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--exp_name', type=str, default='test') ### wandb 실험 제목, pth 저장하는 폴더 이름
    parser.add_argument('--CosineAnealing', type=bool, default=False)
    parser.add_argument('--validation', type=bool, default=False)
    parser.add_argument('--pretrained_path', type=str, default='/opt/ml/code/trained_models/latest.pth')

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')
    return args

def increment_path(model_dir, exp_name, exist_ok=False): ### Custom : 실험명 자동 네이밍 기능
    """ Automatically increment path, i.e. trained_models/exp --> trained_models/exp0, trained_models/exp1 etc.
    Args:
        exist_ok (bool): whether increment path (increment if False).
    """
    path = osp.join(model_dir, exp_name)
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(exp_name)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{exp_name}{n}"

def min_max_bbox(bbox):
    x_list, y_list = [], []
    for point in bbox:
        x_list.append(point[0])
        y_list.append(point[1])
    return [min(x_list), min(y_list), max(x_list), max(y_list)]

def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, exp_name, CosineAnealing, validation, pretrained_path):

    wandb.login()
    exp_name = increment_path(model_dir, exp_name)
    config = args.__dict__
    config['exp_name'] = exp_name
    wandb.init(project="myeongu-data", entity="cv-3-bitcoin",name=exp_name, config=config)
    ### project명 수정 필요합니다!

    if validation: ## validation while training
        print('Validation is applied!')
        dataset = SceneTextDataset(data_dir, split='train_v1', image_size=image_size, crop_size=input_size)
        dataset = EASTDataset(dataset)
    else:
        dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size)
        dataset = EASTDataset(dataset)
    
    ## validation
    if validation:
        with open(osp.join(data_dir, 'ufo/{}.json'.format('valid_v1')), 'r') as f:
            val_anno = json.load(f)
        val_image_fname = sorted(val_anno['images'].keys())
        val_image_dir = osp.join(data_dir, 'images')
        gt_bboxes_dict = {}

        for image_fname in val_image_fname:
            words_info = val_anno['images'][image_fname]['words'].values()
            words_bboxes = [word_info['points'] for word_info in words_info]
            gt_bboxes_dict[image_fname] = [min_max_bbox(bbox) for bbox in words_bboxes]
            # if image_fname in val_anno['images']:
            #     print(gt_bboxes_dict)
            

    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    if pretrained_path:
        print("pretrained mode!")
        model.load_state_dict(torch.load(pretrained_path))
    model.to(device)
    
    if CosineAnealing:
        optimizer = torch.optim.Adam(model.parameters(), lr = 0)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=max_epoch//7, T_mult=2, eta_max=0.1, T_up=10, gamma=0.1)
        print("[Scheduler]: CosineAnnealingWarmUpRestarts is applied!")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    for epoch in range(max_epoch):
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                wandb.log({
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                })
                ### wandb 로깅 부분
                pbar.set_postfix(val_dict)

            ## validation 추가
            if validation:
                model.eval()
                
                pred_bboxes_dict = {}
                val_images = []
                pred_bboxes = []

                with torch.no_grad():
                    for image_fname in val_image_fname:
                        image_fpath = osp.join(val_image_dir, image_fname)
                        val_images.append(cv2.imread(image_fpath)[:, :, ::-1])
                        if len(val_images) == batch_size:
                            pred_bboxes.extend(detect(model, val_images, input_size))
                            val_images = []

                    if len(val_images):
                        pred_bboxes.extend(detect(model, val_images, input_size))

                    for image_fname, bboxes in zip(val_image_fname, pred_bboxes):
                        pred_bboxes_dict[image_fname] = [min_max_bbox(bbox) for bbox in bboxes.tolist()]
                    
                    deteval_metrics = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict)['total']
                    print('[precision=' + str(deteval_metrics['precision'])
                        + ' recall=' + str(deteval_metrics['precision'])
                        + ' hmean=' + str(deteval_metrics['hmean']) + '] \n'
                        )
                    wandb.log(deteval_metrics)
                #################

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            exp_path = osp.join(model_dir, exp_name)
            if not osp.exists(exp_path):
                os.makedirs(exp_path)
            ckpt_fpath = osp.join(exp_path, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)

def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
