import argparse
import json
import os
import pickle
import torch

import cv2
import numpy as np
from pyquaternion.quaternion import Quaternion

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB



def lidar2img(points_lidar, camera2lidar, camera2img):
    points_lidar_homogeneous = \
        np.concatenate([points_lidar,
                        np.ones((points_lidar.shape[0], 1),
                                dtype=points_lidar.dtype)], axis=1)
    # camera2lidar = np.eye(4, dtype=np.float32)
    # camera2lidar[:3, :3] = camrera_info['sensor2lidar_rotation']
    # camera2lidar[:3, 3] = camrera_info['sensor2lidar_translation']
    if isinstance(camera2lidar, torch.Tensor):
        camera2lidar = camera2lidar.cpu().numpy()
    lidar2camera = np.linalg.inv(camera2lidar)
    # import pdb; pdb.set_trace()
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    # camera2img = camrera_info['cam_intrinsic']
    if isinstance(camera2img, torch.Tensor):
        camera2img = camera2img.cpu().numpy()
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid


def get_lidar2global(infos):
    lidar2ego = np.eye(4, dtype=np.float32)
    lidar2ego[:3, :3] = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = infos['lidar2ego_translation']
    ego2global = np.eye(4, dtype=np.float32)
    ego2global[:3, :3] = Quaternion(
        infos['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = infos['ego2global_translation']
    return ego2global @ lidar2ego

def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(
        valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    return valid

def lidar3d_show(pred_bboxes, pred_labels, gt_bboxes=None):
    from mmdet3d.core import Box3DMode, Coord3DMode
    from mmdet3d.core.visualizer.open3d_vis import Visualizer

    points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                        Coord3DMode.DEPTH)
    pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                    Box3DMode.DEPTH)
    pred_bboxes = pred_bboxes.tensor.cpu().numpy()
        
    vis = Visualizer(points)
    if pred_bboxes is not None:
        if pred_labels is None:
            vis.add_bboxes(bbox3d=pred_bboxes)
        else:
            palette = np.random.randint(
                0, 255, size=(pred_labels.max() + 1, 3)) / 256
            labelDict = {}
            for j in range(len(pred_labels)):
                i = int(pred_labels[j].numpy())
                if labelDict.get(i) is None:
                    labelDict[i] = []
                labelDict[i].append(pred_bboxes[j])
            for i in labelDict:
                vis.add_bboxes(
                    bbox3d=np.array(labelDict[i]),
                    bbox_color=palette[i],
                    points_in_box_color=palette[i])

    if gt_bboxes is not None:
        vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 1))
    show_path = 'tmp_online.png'
    vis.show(show_path)


def bbox3d_show():
    pass
