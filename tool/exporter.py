# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, glob
import onnx
import torch
import argparse
import numpy as np

from pathlib import Path
from onnxsim import simplify
from pcdet.utils import common_utils
from pcdet.models import build_network
from pcdet.datasets import DatasetTemplate
from pcdet.config import cfg, cfg_from_yaml_file

from exporter_paramters import export_paramters as export_paramters
from simplifier_onnx import simplify_preprocess, simplify_postprocess

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--max_voxels', type=int, default=25000,
                        help='Max number of voxels (voxel02: 25000, voxel03: 15000,  voxel04: 12500)')
    
    # Choice 1: 자동으로 해당 학습 결과에서 학습 결과와 configuration 불러와서 onnx/ 경로에 저장
    parser.add_argument('--version', type=str, default=None, help='Automatically load config and weights using training version')
    parser.add_argument('--epoch', type=str, default="latest_model", help='Epoch number')
    
    # Choice 2: 기존 방법, 각 파일에 대한 경로 직접 정의
    parser.add_argument('--cfg_file', type=str, default='cfgs/argo2_models/voxel04_pointpillar_v3.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default='/home/ailab/AILabDataset/03_Shared_Repository/jinsu/03_HMG_AVC/OpenPCDet/Real/argo2_models/voxel04_pointpillar_v3/default/ckpt/checkpoint_epoch_200.pth',
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')


    args = parser.parse_args()
    ROOT_DIR = Path("/home/ailab/AILabDataset/03_Shared_Repository/jinsu/03_HMG_AVC/OpenPCDet/Real/argo2_models/")
    
    # Choice 1
    if args.version is not None:
        print('------ Using Auto Mode ------')
        args.cfg_file = str(ROOT_DIR / args.version / "default" / args.version) + ".yaml"
        if args.epoch == "latest_model":
            args.ckpt = str(ROOT_DIR / args.version / "default" / "ckpt" / "latest_model")+ ".pth"
        else:
            args.ckpt = str(ROOT_DIR / args.version / "default" / "ckpt" / "checkpoint_epoch_") + args.epoch + ".pth"

        # a) folder by folder
        # args.savedir = ROOT_DIR / args.version / "default" / "onnx"
        # args.save = ROOT_DIR / args.version / "default" / "onnx" / args.version
        # Example: ROOT_DIR/voxel04_pointpillar_v2/default/onnx/voxel04_pointpillar_v2.onnx
        
        # b) integrated folder
        args.savedir = ROOT_DIR / "onnx"
        args.save = str(ROOT_DIR / "onnx" / args.version) + "_" + args.epoch
        # Example: ROOT_DIR/onnx/voxel04_pointpillar_v2.onnx

        os.makedirs(args.savedir, exist_ok=True)
    # Choice 2
    else:
        print('------ Using Manual Mode ------')
        # 현재 디렉토리에 cfg_file 명과 동일하게 저장
        args.save = Path("onnx") / args.cfg_file.split('/')[-1].split('.')[0]
        os.makedirs("onnx", exist_ok=True) # 출력 결과를 저장할 폴더 생성
    
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    export_paramters(cfg, args)
    logger = common_utils.create_logger()
    logger.info('------ Convert OpenPCDet model for TensorRT ------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    np.set_printoptions(threshold=np.inf)
    with torch.no_grad():

      MAX_VOXELS = args.max_voxels

      NUMBER_OF_CLASSES = len(cfg.CLASS_NAMES)
      MAX_POINTS_PER_VOXEL = None

      DATA_PROCESSOR = cfg.DATA_CONFIG.DATA_PROCESSOR
      POINT_CLOUD_RANGE = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
      for i in DATA_PROCESSOR:
          if i['NAME'] == "transform_points_to_voxels":
              MAX_POINTS_PER_VOXEL = i['MAX_POINTS_PER_VOXEL']
              VOXEL_SIZES = i['VOXEL_SIZE']
              break

      if MAX_POINTS_PER_VOXEL == None:
          logger.info('Could Not Parse Config... Exiting')
          import sys
          sys.exit()

      VOXEL_SIZE_X = abs(POINT_CLOUD_RANGE[0] - POINT_CLOUD_RANGE[3]) / VOXEL_SIZES[0]
      VOXEL_SIZE_Y = abs(POINT_CLOUD_RANGE[1] - POINT_CLOUD_RANGE[4]) / VOXEL_SIZES[1]

    #   assert MAX_VOXELS > VOXEL_SIZE_X * VOXEL_SIZE_Y, "MAX_VOXELS is smaller than VOXEL_NUM! Update MAX_VOXELS"

      FEATURE_SIZE_X = VOXEL_SIZE_X / 2  # Is this number of bins?
      FEATURE_SIZE_Y = VOXEL_SIZE_Y / 2

      dummy_voxels = torch.zeros(
          (MAX_VOXELS, MAX_POINTS_PER_VOXEL, 4),
          dtype=torch.float32,
          device='cuda:0')

      dummy_voxel_idxs = torch.zeros(
          (MAX_VOXELS, 4),
          dtype=torch.int32,
          device='cuda:0')

      dummy_voxel_num = torch.zeros(
          (1),
          dtype=torch.int32,
          device='cuda:0')

      dummy_input = dict()
      dummy_input['voxels'] = dummy_voxels
      dummy_input['voxel_num_points'] = dummy_voxel_num
      dummy_input['voxel_coords'] = dummy_voxel_idxs
      dummy_input['batch_size'] = torch.tensor(1)

      torch.onnx.export(model,       # model being run
          dummy_input,               # model input (or a tuple for multiple inputs)
          str(args.save)+"_raw.onnx",     # where to save the model (can be a file or file-like object)
          export_params=True,        # store the trained parameter weights inside the model file
          opset_version=11,          # the ONNX version to export the model to
          do_constant_folding=True,  # whether to execute constant folding for optimization
          keep_initializers_as_inputs=True,
          input_names = ['voxels', 'voxel_num', 'voxel_idxs'],   # the model's input names
          output_names = ['cls_preds', 'box_preds', 'dir_cls_preds'], # the model's output names
          )

      onnx_raw = onnx.load(str(args.save)+"_raw.onnx")  # load onnx model
      onnx_trim_post = simplify_postprocess(onnx_raw, FEATURE_SIZE_X, FEATURE_SIZE_Y, NUMBER_OF_CLASSES)
      
      onnx_simp, check = simplify(onnx_trim_post)
      assert check, "Simplified ONNX model could not be validated"

      onnx_final = simplify_preprocess(onnx_simp, VOXEL_SIZE_X, VOXEL_SIZE_Y, MAX_POINTS_PER_VOXEL)
      onnx.save(onnx_final, str(args.save)+".onnx")
      print('finished exporting onnx')

    logger.info('[PASS] ONNX EXPORTED: {}'.format(str(args.save)))

if __name__ == '__main__':
    main()