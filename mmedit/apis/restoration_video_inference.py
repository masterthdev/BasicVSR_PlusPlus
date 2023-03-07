# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
import re
from functools import reduce
import os

import mmcv
import numpy as np
import torch
import gc
from mmedit.datasets.pipelines import Compose
from mmedit.core import tensor2img

VIDEO_EXTENSIONS = ('.mp4', '.mov')


def pad_sequence(data, window_size):
    padding = window_size // 2

    data = torch.cat([
        data[:, 1 + padding:1 + 2 * padding].flip(1), data,
        data[:, -1 - 2 * padding:-1 - padding].flip(1)
    ],
                     dim=1)

    return data

def get_batch_count(img_dir):
    
    video_reader = mmcv.VideoReader(img_dir)
    
    import math
    video_size = len(video_reader)
    batch_count = math.ceil(video_size/max_seq_len)
    
    del video_reader
    gc.collect()
    return batch_count

def set_datas(model,batch,max_seq_len,img_dir):
    test_pipeline = 0
    # build the data pipeline
    if model.cfg.get('demo_pipeline', None):
        test_pipeline = model.cfg.demo_pipeline
    elif model.cfg.get('test_pipeline', None):
        test_pipeline = model.cfg.test_pipeline
    else:
        test_pipeline = model.cfg.val_pipeline
    
    batch_begin = batch * max_seq_len
    batch_end = batch_begin + max_seq_len
    # load the images
    data = dict(lq=[], lq_path=None, key=img_dir)
    framno = 0
    video_reader = mmcv.VideoReader(img_dir)
    for frame in video_reader:
        if framno >= batch_begin and framno < batch_end:
            data['lq'].append(np.flip(frame, axis=2))
            print("vidframe " + str(framno))
        framno += 1
    del video_reader
    gc.collect()
    
    # remove the data loading pipeline
    tmp_pipeline = []
    for pipeline in test_pipeline:
        if pipeline['type'] not in [
                'GenerateSegmentIndices', 'LoadImageFromFileList'
        ]:
            tmp_pipeline.append(pipeline)
    test_pipeline = tmp_pipeline
    gc.collect()

    # compose the pipeline
    test_pipeline = Compose(test_pipeline)
    data = test_pipeline(data)
    data = data['lq'].unsqueeze(0)  # in cpu
    gc.collect()

def restoration_video_inference(model,
                                img_dir,
                                window_size,
                                start_idx,
                                filename_tmpl,args,
                                max_seq_len=None):

    device = next(model.parameters()).device  # model device

    batch_count = get_batch_count(img_dir)
    
    for batch in range(batch_count):
        data = set_datas(model,batch,max_seq_len,img_dir)

        # forward the model
        with torch.no_grad():
            for i in range(0, data.size(1), max_seq_len):
                gc.collect()
                output = []
                output.append(
                    model(
                        lq=data[:, i:i + max_seq_len].to(device),
                        test_mode=True)['output'].cpu())
                output = torch.cat(output, dim=1)
                
                gc.collect()
                file_extension = os.path.splitext(args.output_dir)[1]
                for ii in range(args.start_idx, args.start_idx + output.size(1)):
                    output_i = output[:, ii - args.start_idx, :, :, :]
                    output_i = tensor2img(output_i)
                    save_path_i = f'{args.output_dir}/{args.filename_tmpl.format(batch_begin+i+ii)}'

                    mmcv.imwrite(output_i, save_path_i)
                    print("imwrite " + str(batch_begin+i+ii))

    return 
