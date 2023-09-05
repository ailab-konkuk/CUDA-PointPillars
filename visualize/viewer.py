"""
author: hova88
date: 2021/03/16
"""
import numpy as np
from visual_tools import draw_clouds_with_boxes
import open3d as o3d

if __name__ == "__main__":
    
    cloud_path = '../custom_data_1/000010.bin'
    boxes_path = '../custom_data_1/pp_argo2_v3/000010.txt'
    
    cloud = np.fromfile(cloud_path, dtype=np.float32).reshape(-1,4)
    boxes = np.loadtxt(boxes_path).reshape(-1,9)
    boxes = boxes[boxes[:, -1] > 0.2][:, :7] # score thr = 0.8
    draw_clouds_with_boxes(cloud, boxes)