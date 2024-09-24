# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
from tqdm import tqdm
from PIL import Image
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import numpy as np
from predictor import VisualizationDemo
from adet.config import get_cfg
import pandas as pd
# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    # cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    # cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", help="input image or directory")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def get_all_keyframes(root_dir):
    keyframes_dir_list = [f'{root_dir}/{x}/Keyframes' for x in os.listdir(root_dir)]
    all_keyframe_paths = dict()
    for keyframe_dir in keyframes_dir_list:
        for part in sorted(os.listdir(keyframe_dir)):
            data_part = part.split('_')[-2] # L01, L02 for ex
            all_keyframe_paths[data_part] =  dict()
    for keyframe_dir in keyframes_dir_list:
        for data_part in sorted(all_keyframe_paths.keys()):
            data_part_path = f'{keyframe_dir}/{data_part}_extra'
            if os.path.isdir(data_part_path):
                video_dirs = sorted(os.listdir(data_part_path))
                video_ids = [video_dir.split('_')[-1] for video_dir in video_dirs]
                for video_id, video_dir in zip(video_ids, video_dirs):
                    keyframe_paths = sorted(glob.glob(f'{data_part_path}/{video_dir}/*.jpg'))
                    all_keyframe_paths[data_part][video_id] = keyframe_paths

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if args.input:
        # all_keyframe_paths = get_all_keyframes_paths(args.input)
        # for key, video_keyframe_paths in tqdm(all_keyframe_paths.items()):
        #     video_ids = sorted(video_keyframe_paths.keys())
        #     if not os.path.exists(os.path.join(args.output, key)):
        #         os.mkdir(os.path.join(args.output, key))
        #     for video_id in tqdm(video_ids):
        #         video_keyframe_path = video_keyframe_paths[video_id]
        #         for i in tqdm(range(0, len(video_keyframe_path), bs)):
        #             image_paths = video_keyframe_path[i:i+bs]
                    # use PIL, to be consistent with evaluation
        path = args.input
        img = read_image(path, format="RGB")
        predictions, _ = demo.run_on_batch_image([img])
        for prediction in predictions:
            instances = prediction["instances"].to(demo.cpu_device)
            bds = np.asarray(instances.bd)
            bds_bbox= []
            for bd in bds:
                bd = np.hsplit(bd, 2)
                bd = np.vstack([bd[0], bd[1][::-1]])
                bd = np.hsplit(bd, 2)
                _x = bd[0].reshape(-1)
                _y = bd[1].reshape(-1)
                bds_bbox.append([_x,_y])
            bbox = []
            for itr in bds_bbox:
                x_min = min(itr[0])
                x_max = max(itr[0])
                y_min = min(itr[1])
                y_max = max(itr[1])
                bbox.append([x_min,y_min,x_max,y_max])
            pil_img = Image.fromarray(img)
            text = []
            
            for i, box in enumerate(bbox):
                cropped_img = pil_img.crop(box)
                if args.output:
                    if os.path.isdir(args.output):
                        
                        frame_id, ext = os.path.basename(args.output).split('.')
                        basename = f"{i}.{ext}"
                        out_filename = os.path.join(args.output, frame_id, basename)
                        if not os.path.exists(os.path.join(args.output, frame_id)):
                            os.mkdir(os.path.join(args.output, frame_id))
                        cropped_img.save(out_filename)
                    else:
                        raise "Please specify a directory with args.output"
