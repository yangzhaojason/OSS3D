#author jingyu li
from PIL import Image
import copy
import os
import shutil
import numpy as np
import torch
import torchvision.transforms as transforms
from mmcv import is_list_of
from PIL import ImageFilter, ImageDraw
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines import Compose as BaseCompose
from mmdet3d.datasets.pipelines import PrepareImageInputs, LoadAnnotationsBEVDepth, PointToMultiViewDepth
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from pyquaternion import Quaternion

@PIPELINES.register_module()
class MultiBranch(object):
    def __init__(self, **transform_group):
        self.transform_group = {k: BaseCompose(v) for k, v in transform_group.items()}

    def __call__(self, results):
        multi_results = []
        for k, v in self.transform_group.items():
            res = v(copy.deepcopy(results))
            if res is None:
                return None
            #res["img_metas"]["tag"] = k
            multi_results.append(res)
        return multi_results

@PIPELINES.register_module()
class STMultiBranch(object):
    def __init__(self, is_seq=False, **transform_group):
        self.is_seq = is_seq
        self.transform_group = {k: BaseCompose(v) for k, v in transform_group.items()}

    def __call__(self, results):
        multi_results = []
        if self.is_seq:
            weak_pipe = self.transform_group['unsup_weak']
            strong_pipe = self.transform_group['unsup_strong']
            res = weak_pipe(copy.deepcopy(results))
            multi_results.append(res)            
            multi_results.append(strong_pipe(copy.deepcopy(results)))
            #multi_results.append(copy.deepcopy(res))
            #res.pop('tag')
            #multi_results.append(strong_pipe(res))
            for k, v in self.transform_group.items():
                if 'common' in k:
                    for idx in range(len(multi_results)):
                        multi_results[idx] = v(multi_results[idx])
        else:
            for k, v in self.transform_group.items():
                res = v(copy.deepcopy(results))
                if res is None:
                    return None
                multi_results.append(res)
        return multi_results

class DTSingleOperation:
    def __init__(self):
        self.transform = None

    def __call__(self, results):
        tmp = []
        for img in results['canvas']:
            tmp.append(self.transform(img))
        results['canvas'] = tmp
        return results

@PIPELINES.register_module()
class DTRandomGrayscale(DTSingleOperation):
    def __init__(self, p=0.2):
        super(DTRandomGrayscale, self).__init__()
        self.transform = transforms.RandomGrayscale(p=p)

@PIPELINES.register_module()
class DTToPILImage(DTSingleOperation):
    def __init__(self):
        super(DTToPILImage, self).__init__()
        self.transform = transforms.ToPILImage()

@PIPELINES.register_module()
class DTToNumpy:
    def __call__(self, results):
        breakpoint()
        keyresults = results['img_inputs'][0]
        keyresults = np.asarray(keyresults)
        results['img_np'] = keyresults
        return results

@PIPELINES.register_module()
class DTRandomErasing(DTSingleOperation):
    def __init__(self, n_iterations_range=(1, 6), size=(0.02, 0.33), squared=True):
        super(DTRandomErasing, self).__init__()
        self.n_iterations = np.random.randint(*n_iterations_range)
        self.size = size
        self.squared = squared
        self.transform = self.erase_transform

    def erase_transform(self, img):
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for _ in range(self.n_iterations):
            erase_size = np.random.uniform(self.size[0], self.size[1])
            if self.squared:
                erase_h = erase_w = int(h * erase_size)
            else:
                erase_h = int(h * erase_size)
                erase_w = int(w * np.random.uniform(self.size[0], self.size[1]))
            erase_x = np.random.randint(0, w - erase_w)
            erase_y = np.random.randint(0, h - erase_h)
            draw.rectangle([erase_x, erase_y, erase_x + erase_w, erase_y + erase_h], fill=(0, 0, 0))
        return img
    
@PIPELINES.register_module()
class DTToTensor:
    def __init__(self, before=True):
        self.before = before
    def __call__(self, results ):
        imgs = results['canvas']
        newimgs = []
        for img in imgs:
            newimgs.append(mmlabNormalize(img))
        keyresults = torch.stack(newimgs)
        #keyresults = torch.tensor(imgs)
        if self.before:
            imgs, rots, trans, camera2lidars = results['img_inputs'][:4]
            intrins, post_rots, post_trans, bda_rot = results['img_inputs'][4:]
            results['img_inputs'] = (keyresults, rots, trans, camera2lidars, intrins, post_rots,
                                     post_trans, bda_rot)
        else:
            imgs, rots, trans, camera2lidars = results['img_inputs'][:4]
            intrins, post_rots, post_trans = results['img_inputs'][4:]
            results['img_inputs'] = (keyresults, rots, trans, camera2lidars, intrins, post_rots,
                                     post_trans )
        return results

@PIPELINES.register_module()
class DTRandomApply:
    def __init__(self, operations, p=0.5):
        self.p = p
        if is_list_of(operations, dict):
            self.operations = []
            for ope in operations:
                self.operations.append(build_dt_aug(**ope))
        else:
            self.operations = operations

    def __call__(self, results):
        if self.p < np.random.random():
            return results
        imgs = results['canvas']
        tmp = []
        for img in imgs:
            for ope in self.operations:
                img = ope(img)
            tmp.append(img)
        results['canvas'] = tmp
        return results

class DTGaussianBlur:
    def __init__(self, rad_range=[0.1, 2.0]):
        self.rad_range = rad_range

    def __call__(self, x):
        rad = np.random.uniform(*self.rad_range)
        x = x.filter(ImageFilter.GaussianBlur(radius=rad))
        return x

DT_LOCAL_AUGS = {
    'DTGaussianBlur': DTGaussianBlur
}

@PIPELINES.register_module()
class Show_Image:
    def __init__(self, save_path='/data1/vis', isstrong=False):
        self.save_path = save_path
        self.isstrong = isstrong
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)
    def __call__(self, results):
        imgs = results['canvas']
        breakpoint()
        topil = transforms.ToPILImage()
        if not self.isstrong:
            for i in range(6):
                img = imgs[i]
                img = topil(img)
                img.save(os.path.join(self.save_path, f"weak_{i}.jpg"))
        else:
            for i in range(6):
                img = imgs[i]
                img = topil(img)
                img.save(os.path.join(self.save_path, f"strong_{i}.jpg"))

        return results


def build_dt_aug(type, **kwargs):
    return DT_LOCAL_AUGS[type](**kwargs)

def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img

prev= dict()
@PIPELINES.register_module()
class LoadAnnotationsBEVSemi(object):
    def __init__(self, bda_aug_conf, classes, is_init=False):
        self.bda_aug_conf = bda_aug_conf
        self.is_init = is_init
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_init:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']

            prev['rotate_bda'] = rotate_bda
            prev['scale_bda'] = scale_bda
            prev['flip_dx'] = flip_dx
            prev['flip_dy'] = flip_dy
        else:
            rotate_bda = prev['rotate_bda']
            scale_bda = prev['scale_bda']
            flip_dx = prev['flip_dx']
            flip_dy = prev['flip_dy']
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat
    
    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
                                 post_trans, bda_rot)
        results['bdaiso'] = (rotate_bda,scale_bda,flip_dx, flip_dy)
        if 'voxel_semantics' in results:
            if flip_dx:
                results['voxel_semantics'] = results['voxel_semantics'][::-1,...].copy()
                results['mask_lidar'] = results['mask_lidar'][::-1,...].copy()
                results['mask_camera'] = results['mask_camera'][::-1,...].copy()
            if flip_dy:
                results['voxel_semantics'] = results['voxel_semantics'][:,::-1,...].copy()
                results['mask_lidar'] = results['mask_lidar'][:,::-1,...].copy()
                results['mask_camera'] = results['mask_camera'][:,::-1,...].copy()
        return results

@PIPELINES.register_module()
class ExtraAttrs(object):
    def __init__(self, **attrs):
        self.attrs = attrs

    def __call__(self, results):
        for k, v in self.attrs.items():
            assert k not in results
            results[k] = v
        return results

@PIPELINES.register_module()
class PrepareImageInputsBEVDepth(PrepareImageInputs):

        def get_inputs(self, results, flip=None, scale=None):
            imgs = []
            sensor2egos = []
            ego2globals = []
            camera2lidars = []
            filenames = []
            intrins = []
            post_rots = []
            post_trans = []
            cam_names = self.choose_cams()
            results['cam_names'] = cam_names
            canvas = []
            for cam_name in cam_names:
                cam_data = results['curr']['cams'][cam_name]
                filename = cam_data['data_path']
                img = Image.open(filename)
                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                intrin = torch.Tensor(cam_data['cam_intrinsic'])

                camera2lidar = torch.eye(4)
                camera2lidar[:3, :3] = torch.Tensor(cam_data['sensor2lidar_rotation'])
                camera2lidar[:3, 3] = torch.Tensor(cam_data['sensor2lidar_translation'])

                sensor2ego, ego2global = \
                    self.get_sensor_transforms(results['curr'], cam_name)
                # image view augmentation (resize, crop, horizontal flip, rotate)
                img_augs = self.sample_augmentation(
                    H=img.height, W=img.width, flip=flip, scale=scale)
                resize, resize_dims, crop, flip, rotate = img_augs
                try:
                    img, post_rot2, post_tran2 = \
                        self.img_transform(img, post_rot,
                                        post_tran,
                                        resize=resize,
                                        resize_dims=resize_dims,
                                        crop=crop,
                                        flip=flip,
                                        rotate=rotate)
                except Exception as e:
                    # 捕获异常，e 包含了异常的详细信息
                    print(f"An error occurred while processing the image: {str(e)}")
                    print(f"The problematic image path is: {filename}")

                # for convenience, make augmentation matrices 3x3
                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                canvas.append(np.array(img))
                imgs.append(self.normalize_img(img))
                filenames.append(filename)

                if self.sequential:
                    assert 'adjacent' in results
                    for adj_info in results['adjacent']:
                        filename_adj = adj_info['cams'][cam_name]['data_path']
                        img_adjacent = Image.open(filename_adj)
                        img_adjacent = self.img_transform_core(
                            img_adjacent,
                            resize_dims=resize_dims,
                            crop=crop,
                            flip=flip,
                            rotate=rotate)
                        imgs.append(self.normalize_img(img_adjacent))
                intrins.append(intrin)
                sensor2egos.append(sensor2ego)
                ego2globals.append(ego2global)
                camera2lidars.append(camera2lidar)
                post_rots.append(post_rot)
                post_trans.append(post_tran)

            if self.sequential:
                for adj_info in results['adjacent']:
                    post_trans.extend(post_trans[:len(cam_names)])
                    post_rots.extend(post_rots[:len(cam_names)])
                    intrins.extend(intrins[:len(cam_names)])

                    # align
                    for cam_name in cam_names:
                        sensor2ego, ego2global = \
                            self.get_sensor_transforms(adj_info, cam_name)
                        sensor2egos.append(sensor2ego)
                        ego2globals.append(ego2global)

            imgs = torch.stack(imgs) # (Ncams, 3, H, W)
            
            if imgs.shape[0] < len(
                self.data_config['cams']):
                # 用0将imgs的相机数量补到self.data_config['cams']的大小
                imgs = torch.cat(
                    [imgs, torch.zeros(
                        len(self.data_config['cams']) - imgs.shape[0],
                        *imgs.shape[1:])])
                canvas.append(np.zeros_like(canvas[0]))
                sensor2egos.append(torch.eye(4))
                ego2globals.append(torch.eye(4))
                camera2lidars.append(torch.eye(4))
                intrins.append(torch.eye(3))
                post_rots.append(torch.eye(3))
                post_trans.append(torch.zeros(3))

            sensor2egos = torch.stack(sensor2egos)
            ego2globals = torch.stack(ego2globals)
            camera2lidars = torch.stack(camera2lidars)
            intrins = torch.stack(intrins)
            post_rots = torch.stack(post_rots)
            post_trans = torch.stack(post_trans)
            results['canvas'] = canvas
            results['filename'] = filenames
            return (imgs, sensor2egos, ego2globals, camera2lidars, intrins, post_rots, post_trans)

@PIPELINES.register_module()
class LoadAnnotationsBEVDepth_v1(LoadAnnotationsBEVDepth):
        
        def __call__(self, results):
            gt_boxes, gt_labels = results['ann_infos']
            gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
            rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
            bda_mat = torch.zeros(4, 4)
            bda_mat[3, 3] = 1
            gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                                flip_dx, flip_dy)
            bda_mat[:3, :3] = bda_rot
            if len(gt_boxes) == 0:
                gt_boxes = torch.zeros(0, 9)
            results['gt_bboxes_3d'] = \
                LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                    origin=(0.5, 0.5, 0.5))
            results['gt_labels_3d'] = gt_labels
            imgs, rots, trans, camera2lidars= results['img_inputs'][:4]
            intrins, post_rots, post_trans = results['img_inputs'][4:]
            results['img_inputs'] = (imgs, rots, trans, camera2lidars, intrins, post_rots,
                                    post_trans, bda_rot)
            results['bdaiso'] = (rotate_bda,scale_bda,flip_dx, flip_dy)
            if 'voxel_semantics' in results:
                if flip_dx:
                    results['voxel_semantics'] = results['voxel_semantics'][::-1,...].copy()
                    results['mask_lidar'] = results['mask_lidar'][::-1,...].copy()
                    results['mask_camera'] = results['mask_camera'][::-1,...].copy()
                if flip_dy:
                    results['voxel_semantics'] = results['voxel_semantics'][:,::-1,...].copy()
                    results['mask_lidar'] = results['mask_lidar'][:,::-1,...].copy()
                    results['mask_camera'] = results['mask_camera'][:,::-1,...].copy()
            return results

@PIPELINES.register_module()
class PointToMultiViewDepth_Semi(PointToMultiViewDepth):

    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, camera2lidars = results['img_inputs'][:4]
        intrins, post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                             imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results

# @PIPELINES.register_module()
# class LoadOccGTFromFile(object):
#     def __call__(self, results):
#         occ_gt_path = results['occ_gt_path']
#         occ_gt_path = os.path.join(occ_gt_path, "labels.npz")

#         occ_labels = np.load(occ_gt_path)
#         semantics = occ_labels['semantics']
#         mask_lidar = occ_labels['mask_lidar']
#         mask_camera = occ_labels['mask_camera']

#         results['voxel_semantics'] = semantics
#         results['mask_lidar'] = mask_lidar
#         results['mask_camera'] = mask_camera
#         return results
    

# @PIPELINES.register_module()
# class LoadAnnotations(object):

#     def __call__(self, results):
#         gt_boxes, gt_labels = results['ann_infos']
#         gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
#         if len(gt_boxes) == 0:
#             gt_boxes = torch.zeros(0, 9)
#         results['gt_bboxes_3d'] = \
#             LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
#                                  origin=(0.5, 0.5, 0.5))
#         results['gt_labels_3d'] = gt_labels
#         return results


# @PIPELINES.register_module()
# class BEVAug(object):

#     def __init__(self, bda_aug_conf, classes, is_train=True):
#         self.bda_aug_conf = bda_aug_conf
#         self.is_train = is_train
#         self.classes = classes

#     def sample_bda_augmentation(self):
#         """Generate bda augmentation values based on bda_config."""
#         if self.is_train:
#             rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
#             scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
#             flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
#             flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
#             translation_std = self.bda_aug_conf.get('tran_lim', [0.0, 0.0, 0.0])
#             tran_bda = np.random.normal(scale=translation_std, size=3).T
#         else:
#             rotate_bda = 0
#             scale_bda = 1.0
#             flip_dx = False
#             flip_dy = False
#             tran_bda = np.zeros((1, 3), dtype=np.float32)
#         return rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda

#     def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
#                       flip_dy, tran_bda):
#         rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
#         rot_sin = torch.sin(rotate_angle)
#         rot_cos = torch.cos(rotate_angle)
#         rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
#                                 [0, 0, 1]])
#         scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
#                                   [0, 0, scale_ratio]])
#         flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#         if flip_dx:
#             flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
#                                                 [0, 0, 1]])
#         if flip_dy:
#             flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
#                                                 [0, 0, 1]])
#         rot_mat = flip_mat @ (scale_mat @ rot_mat)
#         if gt_boxes.shape[0] > 0:
#             gt_boxes[:, :3] = (
#                 rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
#             gt_boxes[:, 3:6] *= scale_ratio
#             gt_boxes[:, 6] += rotate_angle
#             if flip_dx:
#                 gt_boxes[:,
#                          6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
#                                                                            6]
#             if flip_dy:
#                 gt_boxes[:, 6] = -gt_boxes[:, 6]
#             gt_boxes[:, 7:] = (
#                 rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
#             gt_boxes[:, :3] = gt_boxes[:, :3] + tran_bda
#         return gt_boxes, rot_mat

#     def __call__(self, results):
#         gt_boxes = results['gt_bboxes_3d'].tensor
#         gt_boxes[:,2] = gt_boxes[:,2] + 0.5*gt_boxes[:,5]
#         rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda = \
#             self.sample_bda_augmentation()
#         bda_mat = torch.zeros(4, 4)
#         bda_mat[3, 3] = 1
#         gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
#                                                flip_dx, flip_dy, tran_bda)
#         if 'points' in results:
#             points = results['points'].tensor
#             points_aug = (bda_rot @ points[:, :3].unsqueeze(-1)).squeeze(-1)
#             points[:,:3] = points_aug + tran_bda
#             points = results['points'].new_point(points)
#             results['points'] = points
#         bda_mat[:3, :3] = bda_rot
#         bda_mat[:3, 3] = torch.from_numpy(tran_bda)
#         if len(gt_boxes) == 0:
#             gt_boxes = torch.zeros(0, 9)
#         results['gt_bboxes_3d'] = \
#             LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
#                                  origin=(0.5, 0.5, 0.5))
#         if 'img_inputs' in results:
#             imgs, rots, trans, intrins = results['img_inputs'][:4]
#             post_rots, post_trans = results['img_inputs'][4:]
#             results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
#                                      post_trans, bda_mat)
#         if 'voxel_semantics' in results:
#             if flip_dx:
#                 results['voxel_semantics'] = results['voxel_semantics'][::-1,...].copy()
#                 results['mask_lidar'] = results['mask_lidar'][::-1,...].copy()
#                 results['mask_camera'] = results['mask_camera'][::-1,...].copy()
#             if flip_dy:
#                 results['voxel_semantics'] = results['voxel_semantics'][:,::-1,...].copy()
#                 results['mask_lidar'] = results['mask_lidar'][:,::-1,...].copy()
#                 results['mask_camera'] = results['mask_camera'][:,::-1,...].copy()
#         return results