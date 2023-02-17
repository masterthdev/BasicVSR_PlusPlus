# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import cv2
import mmcv
import numpy as np
import torch

from mmedit.apis import init_model
from mmedit.core import tensor2img
from mmedit.utils import modify_args
from mmedit.datasets.pipelines import Compose

VIDEO_EXTENSIONS = ('.mp4', '.mov')
import glob
import os.path as osp
import re
from functools import reduce

def pad_sequence(data, window_size):
    padding = window_size // 2

    data = torch.cat([
        data[:, 1 + padding:1 + 2 * padding].flip(1), data,
        data[:, -1 - 2 * padding:-1 - padding].flip(1)
    ],
                     dim=1)

    return data
 
def restoration_video_inference(model,
                                img_dir,
                                window_size,
                                start_idx,
                                filename_tmpl,
                                max_seq_len=None,args):

    device = next(model.parameters()).device  # model device

    # build the data pipeline
    if model.cfg.get('demo_pipeline', None):
        test_pipeline = model.cfg.demo_pipeline
    elif model.cfg.get('test_pipeline', None):
        test_pipeline = model.cfg.test_pipeline
    else:
        test_pipeline = model.cfg.val_pipeline

    # check if the input is a video
    file_extension = osp.splitext(img_dir)[1]
    if file_extension in VIDEO_EXTENSIONS:
        video_reader = mmcv.VideoReader(img_dir)
        # load the images
        data = dict(lq=[], lq_path=None, key=img_dir)
        for frame in video_reader:
            data['lq'].append(np.flip(frame, axis=2))

        # remove the data loading pipeline
        tmp_pipeline = []
        for pipeline in test_pipeline:
            if pipeline['type'] not in [
                    'GenerateSegmentIndices', 'LoadImageFromFileList'
            ]:
                tmp_pipeline.append(pipeline)
        test_pipeline = tmp_pipeline
    else:
        # the first element in the pipeline must be 'GenerateSegmentIndices'
        if test_pipeline[0]['type'] != 'GenerateSegmentIndices':
            raise TypeError('The first element in the pipeline must be '
                            f'"GenerateSegmentIndices", but got '
                            f'"{test_pipeline[0]["type"]}".')

        # specify start_idx and filename_tmpl
        test_pipeline[0]['start_idx'] = start_idx
        test_pipeline[0]['filename_tmpl'] = filename_tmpl

        # prepare data
        sequence_length = len(glob.glob(osp.join(img_dir, '*')))
        img_dir_split = re.split(r'[\\/]', img_dir)
        key = img_dir_split[-1]
        lq_folder = reduce(osp.join, img_dir_split[:-1])
        data = dict(
            lq_path=lq_folder,
            gt_path='',
            key=key,
            sequence_length=sequence_length)

    # compose the pipeline
    test_pipeline = Compose(test_pipeline)
    data = test_pipeline(data)
    data = data['lq'].unsqueeze(0)  # in cpu

    # forward the model
    with torch.no_grad():
        if window_size > 0:  # sliding window framework
            data = pad_sequence(data, window_size)
            result = []
            for i in range(0, data.size(1) - 2 * (window_size // 2)):
                data_i = data[:, i:i + window_size].to(device)
                result.append(model(lq=data_i, test_mode=True)['output'].cpu())
            result = torch.stack(result, dim=1)
        else:  # recurrent framework
            if max_seq_len is None:
                result = model(
                    lq=data.to(device), test_mode=True)['output'].cpu()
            else:
                for i in range(0, data.size(1), max_seq_len):
                    output = []
                    output.append(
                        model(
                            lq=data[:, i:i + max_seq_len].to(device),
                            test_mode=True)['output'].cpu())
                    output = torch.cat(result, dim=1)
                    
                    file_extension = os.path.splitext(args.output_dir)[1]
                    #for i in range(args.start_idx, args.start_idx + output.size(1)):
                    output_i = output[:, i - args.start_idx, :, :, :]
                    output_i = tensor2img(output_i)
                    save_path_i = f'{args.output_dir}/{args.filename_tmpl.format(i)}'

                    mmcv.imwrite(output_i, save_path_i)
                    
    return 

def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_dir', help='directory of the input video')
    parser.add_argument('output_dir', help='directory of the output video')
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='index corresponds to the first frame of the sequence')
    parser.add_argument(
        '--filename-tmpl',
        default='{:08d}.png',
        help='template of the file names')
    parser.add_argument(
        '--window-size',
        type=int,
        default=0,
        help='window size if sliding-window framework is used')
    parser.add_argument(
        '--max-seq-len',
        type=int,
        default=None,
        help='maximum sequence length if recurrent framework is used')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    """ Demo for video restoration models.

    Note that we accept video as input/output, when 'input_dir'/'output_dir'
    is set to the path to the video. But using videos introduces video
    compression, which lowers the visual quality. If you want actual quality,
    please save them as separate images (.png).
    """

    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    
    
    restoration_video_inference(model, args.input_dir,
                                         args.window_size, args.start_idx,
                                         args.filename_tmpl, args.max_seq_len, args)
    
    


if __name__ == '__main__':
    main()
