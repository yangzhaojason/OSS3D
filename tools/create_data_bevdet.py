# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import pickle
import mmcv
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from os import path as osp

from tools.data_converter import nuscenes_converter as nuscenes_converter

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def get_gt(info):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels


def nuscenes_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)


def add_ann_adj_info(extra_tag, nuscenes_version='v1.0-mini', dataroot='./data/nuscenes/'):
    nuscenes = NuScenes(nuscenes_version, dataroot)
    for set in ['train', 'val']:
        dataset = pickle.load(
            open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])
            ann_infos = list()
            for ann in sample['anns']:
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)
            dataset['infos'][id]['ann_infos'] = ann_infos
            dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
            dataset['infos'][id]['scene_token'] = sample['scene_token']

            scene = nuscenes.get('scene', sample['scene_token'])
            dataset['infos'][id]['occ_path'] = \
                './data/nuscenes/gts/%s/%s'%(scene['name'], info['token'])
        with open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set),
                  'wb') as fid:
            pickle.dump(dataset, fid)


def generate_semi_info(extra_tag, root_path, ratio, percent, version='v1.0-mini'):
    labeled_info = []
    unlabeled_info = []
    metadata = dict(version=version)
    for set in ['train']:
        dataset = pickle.load(
            open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))

            info = dataset['infos'][id]
            if id %  ratio == 0:
                labeled_info.append(info)
            else:
                unlabeled_info.append(info)
            
        print('labeled sample: {}, unlabeled sample: {}'.format(
            len(labeled_info), len(unlabeled_info)))
        data = dict(infos=labeled_info, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_labeled_{}.pkl'.format(extra_tag, percent))
        mmcv.dump(data, info_path)
        data['infos'] = unlabeled_info
        info_un_path = osp.join(root_path,
                                '{}_infos_unlabeled_{}.pkl'.format(extra_tag, 100-percent))
        mmcv.dump(data, info_un_path)


def _build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "nuScenes data preparation utilities for BEV-based 3D detection.\n\n"
            "This script supports (based on the original __main__ block, incl. comments):\n"
            "1) Create nuScenes infos (nuscenes_data_prep)\n"
            "2) Add ann_infos/gt and occ_path fields (add_ann_adj_info)\n"
            "3) Split labeled/unlabeled pkl for semi-supervised training (generate_semi_info)\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- prep ----
    prep = subparsers.add_parser(
        "prep",
        help="Create nuScenes info pkl files via nuscenes_converter.create_nuscenes_infos",
    )
    prep.add_argument(
        "--root-path",
        default="./data/nuscenes",
        help="Dataset root path (default: ./data/nuscenes)",
    )
    prep.add_argument(
        "--extra-tag",
        default="nuscenes",
        help="Info prefix / extra tag (default: nuscenes)",
    )
    prep.add_argument(
        "--version",
        default="v1.0-trainval",
        help="nuScenes version string (default: v1.0-trainval)",
    )
    prep.add_argument(
        "--max-sweeps",
        type=int,
        default=0,
        help="Max sweeps for create_nuscenes_infos (default: 0, as in the original comment block)",
    )

    # ---- add_ann ----
    add_ann = subparsers.add_parser(
        "add_ann",
        help="Load *_infos_{train,val}.pkl and add ann_infos/gt + scene_token + occ_path",
    )
    add_ann.add_argument(
        "--extra-tag",
        default="nuscenes",
        help="Extra tag used in pkl filenames (default: nuscenes)",
    )
    add_ann.add_argument(
        "--nuscenes-version",
        default="v1.0-trainval",
        help="nuScenes version passed to NuScenes() (default: v1.0-trainval)",
    )
    add_ann.add_argument(
        "--dataroot",
        default="./data/nuscenes/",
        help="nuScenes dataroot (default: ./data/nuscenes/)",
    )

    # ---- semi ----
    semi = subparsers.add_parser(
        "semi",
        help="Generate labeled/unlabeled info pkls for semi-supervised training",
    )
    semi.add_argument(
        "--extra-tag",
        default="nuscenes",
        help="Extra tag used in pkl filenames (default: nuscenes)",
    )
    semi.add_argument(
        "--root-path",
        default="./data/nuscenes",
        help="Output root path for labeled/unlabeled pkls (default: ./data/nuscenes)",
    )
    semi.add_argument(
        "--ratio",
        type=int,
        required=True,
        help=(
            "Label every N-th sample (id % ratio == 0).\n"
            "Examples from the original comments:\n"
            "  ratio=10 -> percent=10\n"
            "  ratio=5  -> percent=20\n"
            "  ratio=3  -> percent=30"
        ),
    )
    semi.add_argument(
        "--percent",
        type=int,
        required=True,
        help="Labeled percent used in output filename (e.g. 10/20/30). Unlabeled is (100 - percent).",
    )
    semi.add_argument(
        "--version",
        default="v1.0-trainval",
        help="Metadata version stored in output pkl (default: v1.0-trainval)",
    )

    return parser


def main():
    parser = _build_argparser()
    args = parser.parse_args()

    if args.command == "prep":
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            max_sweeps=args.max_sweeps,
        )
    elif args.command == "add_ann":
        print("add_ann_infos")
        add_ann_adj_info(
            args.extra_tag,
            nuscenes_version=args.nuscenes_version,
            dataroot=args.dataroot,
        )
    elif args.command == "semi":
        generate_semi_info(
            args.extra_tag,
            args.root_path,
            args.ratio,
            args.percent,
            version=args.version,
        )
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == '__main__':    
    main()
