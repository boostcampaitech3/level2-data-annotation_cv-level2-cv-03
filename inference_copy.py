import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm

from detect import detect
import wandb



CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/annotated/images'))
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', '/opt/ml/code/trained_models'))
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=20)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size):
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, by_sample_bboxes = [], []
    gt_bboxes_dict = {}
    ##배치 사이즈 만큼만 뽑아서 inference 진행하자
    ##뽑는법 valid_v2에서 파일 명들을 뽑아서 그 중에 batchsize만큼만 뽑음
    my_anno = "/opt/ml/input/data/annotated/ufo/annotation.json"
    annotations = read_json(my_anno)

    for i in annotations['images'].keys():
        image_fnames.append(i)

    images = []
    count = 0
    for i in tqdm(image_fnames):
        filepath = "/opt/ml/input/data/annotated/images/" + i
        images.append(cv2.imread(filepath)[:, :, ::-1])
        count += 1
        if count >=200:
            break
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))
    
    count = 0

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        count += 1
        if count >=200:
            break
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)



    
    return ufo_result


def main(args):
    # Initialize model
    model = EAST(pretrained=False).to(args.device)

    # Get paths to checkpoint files
    ckpt_fpath = osp.join(args.model_dir, 'latest.pth')

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')

    ufo_result = dict(images=dict())
    split_result = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                                    args.batch_size)
    ufo_result['images'].update(split_result['images'])

    output_fname = 'output.csv'
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)
