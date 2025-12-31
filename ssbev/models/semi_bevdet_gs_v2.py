##version 1.0 for semi-bev
## semi-loss for foreground object  with consistency losses and  depth 
import os
import torch
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import cv2
import math
import sklearn.mixture as skm

from mmdet.models import DETECTORS
from mmdet3d.models.builder import build_detector
# from mmdet3d.models.builder import DETECTORS, build_detector
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmdet3d.core import circle_nms
# from .semi_detector import semi_detector
# from mmdet3d.models.detectors import semi_detector
from ssbev.models.detectors import semi_detector
from ..utils.gaussian_sampler import center_to_corner_box2d, calculate_box_mask_gaussian
from ..utils.torch_dist import reduce_mean

_POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
_VOXEL_SIZE = [0.1, 0.1, 0.2]
_OUT_SIZE_FACTOR = 8
views = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
    'CAM_BACK', 'CAM_BACK_RIGHT'
]

draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]
draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                               (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
                               (2, 6), (3, 7)]


@DETECTORS.register_module()
class semi_bevdet_gs_v2(semi_detector):
    def __init__(self, model, train_cfg=None, test_cfg=None):
        super(semi_bevdet_gs_v2, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = train_cfg.get("unsup_weight", 1.0)
            self.sup_weight = train_cfg.get("sup_weight", 1.0)
            self.iter_count = train_cfg.get("iter", 0)
            self.burn_in_steps = train_cfg.get("burn_in_steps", 3200)
            self.weight_suppress = train_cfg.get("weight_suppress", "linear")
            self.thresh = [0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
            if train_cfg.get("use_history_box_fusion", False):
                self.pseudo_label_bank = {}
                self.max_pseudo_label_num = train_cfg.get("history_capacity", 10)
            if train_cfg.get("adaptive_threshold", False):
                self.perform_adaptive_threshold = True
                self.distance_bins = train_cfg.get("distance_bins", [0, 20, 40, np.inf])
                self.buffer_length = train_cfg.get("buffer_length", 100)
                self.score_memory_bank = {}
                for i in range(len(self.distance_bins)):
                    self.score_memory_bank[i] = []
            else:
                self.perform_adaptive_threshold = False

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 8
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, camera2lidars, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        return imgs, [sensor2keyegos, ego2globals, camera2lidars, intrins,
                      post_rots, post_trans, bda]

    def bda_transform_reverse(self, gt_bboxes, rotate_bda, scale_bda, flip_dx, flip_dy):
        res_bboxes = gt_bboxes.detach().clone()

        rotate_angle = torch.tensor(rotate_bda / 180 * np.pi)
        rot_sin = torch.sin(-rotate_angle)
        rot_cos = torch.cos(-rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        scale_mat = torch.Tensor([[1 / scale_bda, 0, 0], [0, 1 / scale_bda, 0], [0, 0, 1 / scale_bda]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        inv_rot_mat = rot_mat @ (scale_mat @ flip_mat)
        inv_rot_mat = inv_rot_mat.cuda()
        if flip_dy:
            res_bboxes[:, 6] = -res_bboxes[:, 6]
        if flip_dx:
            res_bboxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - res_bboxes[:, 6]

        # Compute the inverse rotation matrix
        # inv_rot_mat = rot_mat.transpose(0, 1)
        # Apply the inverse rotation
        res_bboxes[:, :3] = (inv_rot_mat @ res_bboxes[:, :3].unsqueeze(-1)).squeeze(-1)
        res_bboxes[:, 3:6] /= scale_bda
        res_bboxes[:, 6] -= rotate_bda

        return res_bboxes

    def bda_transform(self, gt_bboxes, rotate_bda, scale_ratio, flip_dx, flip_dy):
        res_bboxes = gt_bboxes.detach().clone()

        rotate_angle = torch.tensor(rotate_bda / 180 * np.pi)
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
        rot_mat = rot_mat.cuda()
        if res_bboxes.shape[0] > 0:
            res_bboxes[:, :3] = (
                    rot_mat @ res_bboxes[:, :3].unsqueeze(-1)).squeeze(-1)
            res_bboxes[:, 3:6] *= scale_ratio
            res_bboxes[:, 6] += rotate_angle
            if flip_dx:
                res_bboxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - res_bboxes[:, 6]
            if flip_dy:
                res_bboxes[:, 6] = -res_bboxes[:, 6]
            res_bboxes[:, 7:] = (
                    rot_mat[:2, :2] @ res_bboxes[:, 7:].unsqueeze(-1)).squeeze(-1)

        return res_bboxes

    def forward_train(self, img_inputs=None, img_metas=None, **kwargs):
        super(semi_bevdet_gs_v2, self).forward_train(img_inputs, img_metas, **kwargs)
        ##img_inputs len:7     [b,6,3,h,w]
        gt_bboxes_3d = kwargs.get('gt_bboxes_3d')
        gt_labels_3d = kwargs.get('gt_labels_3d')
        gt_depth = kwargs.get('gt_depth')
        if gt_depth is not None:
            flag_gt_depth = True
        else:
            flag_gt_depth = False
        bdaiso = kwargs.get('bdaiso')
        img, matrix_param = self.prepare_inputs(img_inputs)
        format_data = dict()
        matrix_name = ['sensor2keyegos', 'ego2globals', 'camera2lidars', 'intrins',
                       'post_rots', 'post_trans', 'bda']
        bda_name = ['rotate_bda', 'scale_bda', 'flip_dx', 'flip_dy']
        for idx, img_meta in enumerate(img_metas):
            tag = img_meta['tag']
            if tag not in format_data.keys():
                format_data[tag] = dict()
                format_data[tag]['img_inputs'] = [img[idx]]
                format_data[tag]['img_metas'] = [img_meta]
                format_data[tag]['gt_bboxes_3d'] = [gt_bboxes_3d[idx]]
                format_data[tag]['gt_labels_3d'] = [gt_labels_3d[idx]]
                format_data[tag]['gt_depth'] = [gt_depth[idx]] if flag_gt_depth else []
                for ind, key in enumerate(matrix_name):
                    format_data[tag][key] = [matrix_param[ind][idx]]
                # format_data[tag]['bdaiso'] = dict()
                # DONE: 读取bev增强参数
                for ind, key in enumerate(bda_name):
                   format_data[tag][key] = [bdaiso[ind][idx]]
            else:
                format_data[tag]['img_inputs'].append(img[idx])
                format_data[tag]['img_metas'].append(img_meta)
                format_data[tag]['gt_bboxes_3d'].append(gt_bboxes_3d[idx])
                format_data[tag]['gt_labels_3d'].append(gt_labels_3d[idx])
                if flag_gt_depth:
                    format_data[tag]['gt_depth'].append(gt_depth[idx])
                for ind, key in enumerate(matrix_name):
                    format_data[tag][key].append(matrix_param[ind][idx])
                # DONE: 读取bev增强参数
                for ind, key in enumerate(bda_name):
                   format_data[tag][key].append(bdaiso[ind][idx])
        for tag in format_data.keys():
            format_data[tag]['img_inputs'] = [torch.stack(format_data[tag]['img_inputs'], dim=0)]
            for ind, key in enumerate(matrix_name):
                format_data[tag][key] = torch.stack(format_data[tag][key], dim=0)
                format_data[tag]['img_inputs'].append(format_data[tag][key])
                format_data[tag].pop(key)
            for ind, key in enumerate(bda_name): #DONE: 引入bev增强参数
                format_data[tag][key] = torch.stack(format_data[tag][key], dim=0)
                if 'bda' not in format_data[tag]:
                    format_data[tag]['bda'] = []
                format_data[tag]['bda'].append(format_data[tag][key])
                format_data[tag].pop(key)
        unsup_bs = len(format_data['unsup_teacher']['img_metas'])
        format_data['unsup_teacher']['batch_size'] = unsup_bs
        format_data['unsup_student']['batch_size'] = unsup_bs

        ##remove unsup_student gt
        del format_data['unsup_student']['gt_bboxes_3d']
        del format_data['unsup_student']['gt_labels_3d']
        if flag_gt_depth: del format_data['unsup_student']['gt_depth']

        losses = dict()
        ## supervised weight
        sup_losses = self.student.forward_train(**format_data['sup'])
        for key, val in sup_losses.items():
            if 'loss' in key:
                if isinstance(val, list):
                    losses[f"{key}_sup"] = [self.sup_weight * x for x in val]
                else:
                    losses[f"{key}_sup"] = self.sup_weight * val
            else:
                losses[key] = val
        ## smi-supervised
        if self.iter_count > self.burn_in_steps:
            unsup_weight = self.get_unsup_weight(self.weight_suppress)
            unsup_losses = self.unsup_forward_train(teacher_data=format_data['unsup_teacher'],
                                                    student_data=format_data['unsup_student'])
            for key, val in unsup_losses.items():
                if isinstance(val, dict):
                    for k, v in val.items():
                        losses[f"{key}_{k}_unsup"] = unsup_weight * v
                elif 'loss' in key:
                    if isinstance(val, list):
                        losses[f"{key}_unsup"] = [unsup_weight * x for x in val]
                    else:
                        losses[f"{key}_unsup"] = unsup_weight * val
                else:
                    losses[key] = val
        self.iter_count += 1

        return losses

    def semi_depth_loss(self, depth_labels, depth_preds):
        from torch.cuda.amp.autocast_mode import autocast

        # depth_labels
        # import pdb; pdb.set_trace()
        D = self.student.img_view_transformer.D
        depth_labels = depth_labels.permute(0, 2, 3, 1).contiguous().view(-1, D)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, D)

        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return depth_loss

    def softLabel_loss(self, resp_s, gt_boxes, gt_labels, gt_depth, depth_s, batch_size):
        outs = resp_s
        bs_gt_bboxed = []
        bs_gt_labels = []
        for idx in range(batch_size):
            gt_bboxes_3d = LiDARInstance3DBoxes(gt_boxes[idx], box_dim=gt_boxes[idx].shape[-1])
            bs_gt_bboxed.append(gt_bboxes_3d)
            bs_gt_labels.append(gt_labels[idx])

        loss_inputs = [bs_gt_bboxed, bs_gt_labels, [outs]]
        softlabel_losses = self.student.pts_bbox_head.loss(*loss_inputs)

        # todo add depth-loss
        # import pdb; pdb.set_trace()
        loss_depth = self.semi_depth_loss(gt_depth, depth_s)
        softlabel_losses.update(loss_depth=loss_depth)

        return softlabel_losses

    def image_feature_loss_batch(self, gt_boxes, img_inputs, img_metas, bda_params, student_image_feature,
                                 teacher_image_feature):
        # feature [(bs, 6, C, H, W)]
        from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB
        criterion = nn.MSELoss(reduce=False)
        imgs, sensor2keyegos, ego2globals, camera2lidars, intrins, post_rots, post_trans, bda = img_inputs

        bs = len(gt_boxes)
        assert len(bda_params) == 4
        rotate_bda, scale_bda, flip_dx, flip_dy = bda_params
        image_loss = torch.zeros(1).cuda()
        for idx in range(bs):
            scores_3d = gt_boxes[idx][:, 7]
            labels_3d = gt_boxes[idx][:, 8]
            bboxes_3d = gt_boxes[idx][:, :7]

            res_boxes = self.bda_transform_reverse(bboxes_3d, rotate_bda[idx], scale_bda[idx], flip_dx[idx], flip_dy[idx])

            res_boxes = res_boxes.cpu().detach().numpy()
            boxes = LB(res_boxes, origin=(0.5, 0.5, 0.0))

            imgs_path = img_metas[idx]['filename']
            assert len(imgs[idx]) == len(imgs_path)

            # 原图尺寸大小
            for i in range(len(imgs_path)):
                file_name = imgs_path[i]
                img = cv2.imread(file_name)
                # DONE: 图像的尺寸，box的中心坐标和长宽，因子都设为1；
                boxes_img_h, boxes_img_w, boxes_centers = calc_boxes_imghwc(boxes.corners.numpy(), res_boxes[:, :3],
                                                                            sensor2keyegos[idx][i], intrins[idx][i],
                                                                            img.shape[0], img.shape[1])
                # boxes_img_h (N,), boxes_img_w (N,), boxes_centers (N, 1, 2)
                # 如果非空
                if boxes_img_h.shape[0] > 0:
                    # 拼接成boxes [x,y,0,h,w](centers是二维的，需要补个0)
                    boxes_img = np.concatenate(
                        [boxes_centers[:, 0, :], np.zeros((boxes_centers.shape[0], 1)), boxes_img_h.reshape(-1, 1),
                         boxes_img_w.reshape(-1, 1)], axis=1)
                    boxes_img = np.expand_dims(boxes_img, axis=0)
                    gaussian_mask_image = calculate_box_mask_gaussian(
                        [1, 1, img.shape[0], img.shape[1]],
                        boxes_img,
                        [0, 0, 0, img.shape[0], img.shape[1], 0],
                        [1, 1, 1],
                        1,
                    )
                    # 对mask图做图像同步的数据增强
                    gaussian_mask_image = gaussian_mask_image[0].cpu().detach().numpy()  # (1, H, W)
                    # 根据post_rot和post_trans对mask图进行数据增强
                    # post_rot # (1, 3, 3),  post_trans # (1, 3)
                    transformation_matrix = np.zeros((2, 3))
                    transformation_matrix[:2, :2] = post_rots[idx][i][:2, :2].cpu().detach().numpy()
                    transformation_matrix[:, 2] = post_trans[idx][i][:2].cpu().detach().numpy()
                    aug_gaussian_mask_image = cv2.warpAffine(gaussian_mask_image, transformation_matrix, (
                    img_metas[idx]['batch_input_shape'][1], img_metas[idx]['batch_input_shape'][1]))

                    # 然后resize到特征图的大小
                    target_size = (student_image_feature[0][idx].shape[-2], student_image_feature[0][idx].shape[-1])
                    resize_aug_gaussian_mask_image = cv2.resize(aug_gaussian_mask_image, target_size)

                    # c = student_image_feature[idx].shape[1]
                    weight = resize_aug_gaussian_mask_image.sum()
                    gaussian_mask_ = torch.from_numpy(resize_aug_gaussian_mask_image).transpose(0, 1).cuda()
                    loss_feature = criterion(student_image_feature[0][idx][i] * gaussian_mask_,
                                             teacher_image_feature[0][idx][i] * gaussian_mask_)
                    loss_feature = torch.mean(loss_feature, 0)
                    loss_feature = torch.sum(loss_feature) / (weight + 1e-4)
                    image_loss += loss_feature

        return image_loss

    def get_unsup_weight(self, type):
        unsup_weight = self.unsup_weight
        if self.weight_suppress == 'exp':
            target = self.burn_in_steps + 2000
            if self.iter_count <= target:
                scale = np.exp((self.iter_count - target) / 1000)
                unsup_weight *= scale
        elif self.weight_suppress == 'step':
            target = self.burn_in_steps * 2
            if self.iter_count <= target:
                unsup_weight *= 0.25
        elif self.weight_suppress == 'linear':
            target = self.burn_in_steps * 2
            if self.iter_count <= target:
                unsup_weight *= (self.iter_count - self.burn_in_steps) / self.burn_in_steps
        return unsup_weight

    def unsup_forward_train(self, teacher_data, student_data):
        # test for teacher
        with torch.no_grad():
            # 1. pseudo labels gt_bboxes_3d, gt_labels_3d
            # 2. distill targets: xxx
            teacher_info = self.extract_teacher_info(teacher_data)
            batch_size = teacher_data['batch_size']
        gt_boxes = teacher_info['gt_bboxes_3d']
        gt_labels = teacher_info['gt_labels_3d']
        max_num = max([b.shape[0] for b in gt_boxes])
        bs, box_dims = len(gt_boxes), gt_boxes[0].shape[1]
        gt_boxes_together = torch.zeros((bs, max_num, box_dims), device=gt_boxes[0].device)
        for idx in range(bs):
            num_boxes = gt_boxes[idx].shape[0]
            gt_boxes_together[idx, :num_boxes, :] = gt_boxes[idx]
        # gt_boxes = torch.stack([gt for gt in gt_boxes], dim=0)
        # gt_labels = torch.stack([gl for gl in gt_labels], dim=0)
        # gt_boxes_indice = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1]))
        # for i in range(gt_boxes.shape[0]):
        #     cnt = gt_boxes[i].__len__() - 1
        #     while cnt > 0 and gt_boxes[i][cnt].sum() == 0:
        #         cnt -= 1
        #     gt_boxes_indice[i][: cnt + 1] = 1
        # gt_boxes_indice = gt_boxes_indice.bool()
        # gt_boxes_bev_coords = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1], 4, 2))
        # for i in range(gt_boxes.shape[0]):
        #     gt_boxes_tmp = gt_boxes[i]
        #     gt_boxes_tmp_bev = center_to_corner_box2d(
        #         gt_boxes_tmp[:, :2].cpu().detach().numpy(),
        #         gt_boxes_tmp[:, 3:5].cpu().detach().numpy(),
        #         gt_boxes_tmp[:, 6].cpu().detach().numpy(),
        #         origin=(0.5, 0.5)
        #     )
        #     gt_boxes_bev_coords[i] = gt_boxes_tmp_bev
        # gt_boxes_bev_coords = gt_boxes_bev_coords.cuda()
        # gt_boxes_indice = gt_boxes_indice.cuda()
        # gt_boxes_bev_coords[:, :, :, 0] = (
        #     gt_boxes_bev_coords[:, :, :, 0] - _POINT_CLOUD_RANGE[0]
        # )/(_VOXEL_SIZE[0] * _OUT_SIZE_FACTOR)
        # gt_boxes_bev_coords[:, :, :, 1] = (
        #     gt_boxes_bev_coords[:, :, :, 1] - _POINT_CLOUD_RANGE[1]
        # )/(_VOXEL_SIZE[1] * _OUT_SIZE_FACTOR)

        ## 1.get student boxes (Response Distillation loss)
        student_info = self.extract_student_info(student_data)
        loss_resp_cls, loss_resp_reg, top_mask, gaussian_mask = Responseloss(
            student_info['response'],
            teacher_info['response'],
            gt_boxes_together,
            _POINT_CLOUD_RANGE,
            _VOXEL_SIZE,
            _OUT_SIZE_FACTOR,
        )

        ## 2. soft-label
        softlabel_losses = self.softLabel_loss(
            student_info['response'],
            gt_boxes,
            gt_labels,
            teacher_info['depth'],
            student_info['depth'],
            batch_size
        )

        ## 3. bev features(add mask guide)
        bev_feature_loss = FeatureLoss(
            student_info['features'],
            teacher_info['features'],
            top_mask,
            gaussian_mask
        )

        ## 4. todo object-aware (a. images binary-mask(project) images-feat b.add lidar: Cross-modal Distillation)
        # import pdb; pdb.set_trace()
        # images_loss = images_feature_loss(gt_boxes,
        images_loss = self.image_feature_loss_batch(gt_boxes,
            teacher_data['img_inputs'],
            teacher_data['img_metas'],
            teacher_data['bda'],
            student_info['image_feature'],
            teacher_info['image_feature']
        )

        unsup_losses = dict()
        unsup_losses['loss_resp_cls'] = loss_resp_cls
        unsup_losses['loss_resp_reg'] = loss_resp_reg
        unsup_losses['bev_feature_loss'] = bev_feature_loss
        unsup_losses['softlabel'] = softlabel_losses
        unsup_losses['images_feature_loss'] = images_loss * 0.1

        return unsup_losses

    def extract_student_info(self, student_data):
        student_info={}
        img_feats, _, depth, low_features, image_feature = self.teacher.extract_feat_lowlevelfeatures(points=None,
                                                    img=student_data['img_inputs'],
                                                    img_metas=student_data['img_metas'],
                                                    get_lowlevelfeats=True)
        outs = self.student.pts_bbox_head(img_feats)
        student_info['response'] = outs[0]
        student_info['features'] = img_feats
        student_info['image_feature'] = image_feature
        student_info['depth'] =  depth
        return student_info

    def extract_teacher_info(self, teacher_data):
        teacher_info = {}
        bs = teacher_data['batch_size']
        # img_feats, _, depth, low_features = self.teacher.extract_feat_lowlevelfeatures(points=None,
        #                                                                                img=teacher_data['img_inputs'],
        #                                                                                img_metas=teacher_data[
        #                                                                                    'img_metas'],
        #                                                                                get_lowlevelfeats=True)

        img_feats, _, depth, low_features, image_feature = self.teacher.extract_feat_lowlevelfeatures(points=None,
                                                    img=teacher_data['img_inputs'],
                                                    img_metas=teacher_data['img_metas'],
                                                    get_lowlevelfeats=True)

        # mmdet3d.models.dense_heads.centerpoint_head.CenterHead
        # each task head returns a dict of keys 'reg', 'height', 'dim', 'rot', 'vel', 'heatmap'
        # each feature scale has one;
        # here use single feature scale and single task head so outs[0][0] = {reg, height, dim, rot, vel}
        outs = self.teacher.pts_bbox_head(img_feats)
        # list of batch size elements; each element is a tuple of (boxes, scores, labels)
        proposal_list = self.teacher.pts_bbox_head.get_bboxes(
            outs, teacher_data['img_metas'], rescale=False
        )
        boxes_list = [p[0].to(img_feats[0].device).tensor for p in proposal_list]
        labels_list = [p[2].to(img_feats[0].device) for p in proposal_list]
        scores_list = [p[1].to(img_feats[0].device) for p in proposal_list]

        this_rotate_bda, this_scale_bda, this_flip_dx, this_flip_dy = teacher_data["bda"]

        """
        refine code to support multi-batch
        """
        bs_all_boxes = []
        bs_all_scores = []
        bs_all_labels = []
        for idx in range(bs):
            bs_all_boxes.append([boxes_list[idx]])
            bs_all_scores.append([scores_list[idx]])
            bs_all_labels.append([labels_list[idx]])
        angles = [-90 * np.pi / 180, 90 * np.pi / 180]
        for angle in angles:
            angle = torch.tensor(angle)
            theta = torch.tensor([
                [torch.cos(angle), torch.sin(-angle), 0],
                [torch.sin(angle), torch.cos(angle), 0]
            ], dtype=torch.float).cuda()
            theta = theta.unsqueeze(0)
            theta = theta.repeat(bs, 1, 1)
            # grid = F.affine_grid(theta.unsqueeze(0), img_feats[0].shape)
            # just like cv2 getRotationMatrix2D and warpAffine
            # performs affine transformation of the image feats
            grid = F.affine_grid(theta, img_feats[0].shape)
            img_feats_aug = F.grid_sample(img_feats[0], grid)
            os = self.teacher.pts_bbox_head([img_feats_aug])
            proposal_list = self.teacher.pts_bbox_head.get_bboxes(
                os, teacher_data['img_metas'], rescale=False
            )
            angle = -1 * angle
            for idx in range(bs):
                boxes = proposal_list[idx][0].tensor
                rot_sin = torch.sin(angle)
                rot_cos = torch.cos(angle)
                rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                        [0, 0, 1]]).to(proposal_list[0][0].tensor.device)
                if boxes.shape[0] > 0:
                    boxes[:, :3] = (
                            rot_mat @ boxes[:, :3].unsqueeze(-1)).squeeze(-1)
                    boxes[:, 6] += angle
                bs_all_boxes[idx].append(boxes)
                bs_all_labels[idx].append(proposal_list[idx][2])
                bs_all_scores[idx].append(proposal_list[idx][1])

                # insert pseudo labels
                sample_idx = teacher_data["img_metas"][idx]["sample_idx"]
                if hasattr(self, "pseudo_label_bank"):
                    if sample_idx in self.pseudo_label_bank:
                        history_pseudo_label = self.pseudo_label_bank[sample_idx]

                        for item in history_pseudo_label:
                            rectified_boxes = self.bda_transform_reverse(item["boxes"], *item["bda"])
                            bs_all_boxes[idx].append(self.bda_transform(rectified_boxes,
                                                                        this_rotate_bda[idx],
                                                                        this_scale_bda[idx],
                                                                        this_flip_dx[idx],
                                                                        this_flip_dy[idx]))

                            # bs_all_boxes[idx].append(item["boxes"])

                            bs_all_scores[idx].append(item["scores"])
                            bs_all_labels[idx].append(item["labels"])

        res_boxes = []
        res_scores = []
        res_labels = []
        for idx in range(bs):
            boxes = torch.cat(bs_all_boxes[idx], dim=0)
            labels = torch.cat(bs_all_labels[idx], dim=0)
            scores = torch.cat(bs_all_scores[idx], dim=0)
            # perform nms
            centers = boxes[:, [0, 1]]
            nms_boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
            keep = torch.tensor(
                circle_nms(
                    nms_boxes.detach().cpu().numpy(),
                    self.teacher.test_cfg['pts']['min_radius'][0],
                    post_max_size=self.teacher.test_cfg['pts']['post_max_size']),
                dtype=torch.long,
                device=boxes.device
            )
            res_boxes.append(boxes[keep])
            res_scores.append(scores[keep])
            res_labels.append(labels[keep])

        if self.perform_adaptive_threshold:
            # 划分距离区间，分区间进行阈值筛选
            # 将距离分成不同的区间
            distances = []
            for idx in range(bs):
                distances_idx = torch.norm(res_boxes[idx][:, :3], dim=1)
                distances.append(distances_idx)
            assert len(self.distance_bins) == 4
            # 统计不同区间的分值
            for idx in range(bs):
                for i in range(len(distances[idx])):
                    if distances[idx][i] < self.distance_bins[1]:
                        self.score_memory_bank[0].append(res_scores[idx][i])
                        if len(self.score_memory_bank[0]) > self.buffer_length:
                            self.score_memory_bank[0].pop(0)
                    elif distances[idx][i] < self.distance_bins[2]:
                        self.score_memory_bank[1].append(res_scores[idx][i])
                        if len(self.score_memory_bank[1]) > self.buffer_length:
                            self.score_memory_bank[1].pop(0)
                    else:
                        self.score_memory_bank[2].append(res_scores[idx][i])
                        if len(self.score_memory_bank[2]) > self.buffer_length:
                            self.score_memory_bank[2].pop(0)
            # 转tensor
            if len(self.score_memory_bank[0]) == 100:
                score_list1 = torch.stack(self.score_memory_bank[0])
                gt_tr1 = gmm_policy(score_list1)
            else:
                gt_tr1 = 0.3
            if len(self.score_memory_bank[1]) == 100:
                score_list1 = torch.stack(self.score_memory_bank[1])
                gt_tr2 = gmm_policy(score_list1)
            else:
                gt_tr2 = 0.3
            if len(self.score_memory_bank[2]) == 100:
                score_list1 = torch.stack(self.score_memory_bank[2])
                gt_tr3 = gmm_policy(score_list1)
            else:
                gt_tr3 = 0.3
            # 阈值筛选
            for idx in range(bs):
                mask = torch.zeros_like(res_scores[idx]).bool()
                for i in range(distances[idx].shape[0]):
                    if distances[idx][i] < self.distance_bins[1] and res_scores[idx][i] > gt_tr1:
                        mask[i] = 1
                    elif distances[idx][i] < self.distance_bins[2] and res_scores[idx][i] > gt_tr2:
                        mask[i] = 1
                    else:
                        if res_scores[idx][i] > gt_tr3:
                            mask[i] = 1
                res_boxes[idx] = res_boxes[idx][mask]
                res_labels[idx] = res_labels[idx][mask]
                res_scores[idx] = res_scores[idx][mask]

                # update pseudo label bank
                sample_idx = teacher_data["img_metas"][idx]['sample_idx']
                if hasattr(self, "pseudo_label_bank"):
                    if sample_idx not in self.pseudo_label_bank:
                        self.pseudo_label_bank[sample_idx] = []
                    if len(self.pseudo_label_bank[sample_idx]) == self.max_pseudo_label_num:
                        self.pseudo_label_bank[sample_idx].pop(0)
        # this_rotate_bda, this_scale_bda, this_flip_dx, this_flip_dy = teacher_data["bda"]
                    self.pseudo_label_bank[sample_idx].append({
                        "boxes": res_boxes[idx],
                        "scores": res_scores[idx],
                        "labels": res_labels[idx],
                        "bda": [this_rotate_bda[idx], this_scale_bda[idx], this_flip_dx[idx], this_flip_dy[idx]],
                    })
        else:
            thr = 0.3
            for idx in range(bs):
                mask = res_scores[idx] > thr
                res_boxes[idx] = res_boxes[idx][mask]
                res_labels[idx] = res_labels[idx][mask]
                res_scores[idx] = res_scores[idx][mask]

                # update pseudo label bank
                sample_idx = teacher_data["img_metas"][idx]['sample_idx']
                if hasattr(self, "pseudo_label_bank"):
                    if sample_idx not in self.pseudo_label_bank:
                        self.pseudo_label_bank[sample_idx] = []
                    if len(self.pseudo_label_bank[sample_idx]) == self.max_pseudo_label_num:
                        self.pseudo_label_bank[sample_idx].pop(0)
                    self.pseudo_label_bank[sample_idx].append({
                        "boxes": res_boxes[idx],
                        "scores": res_scores[idx],
                        "labels": res_labels[idx],
                        "bda": [this_rotate_bda[idx], this_scale_bda[idx], this_flip_dx[idx], this_flip_dy[idx]],
                    })

        # import pdb; pdb.set_trace()
        teacher_info['gt_bboxes_3d'] = res_boxes
        teacher_info['gt_labels_3d'] = res_labels
        ## for distill
        teacher_info['response'] = outs[0]
        teacher_info['features'] = img_feats
        teacher_info['depth'] = depth
        teacher_info['image_feature'] = image_feature
        return teacher_info

    def extract_teacher_info_v1(self, teacher_data):
        teacher_info = {}
        img_feats, _, depth, low_features = self.teacher.extract_feat_lowlevelfeatures(points=None,
                                                                                       img=teacher_data['img_inputs'],
                                                                                       img_metas=teacher_data[
                                                                                           'img_metas'],
                                                                                       get_lowlevelfeats=True)
        outs = self.teacher.pts_bbox_head(img_feats)
        proposal_list = self.teacher.pts_bbox_head.get_bboxes_semi(
            outs, teacher_data['img_metas'], rescale=False
        )
        bs = teacher_data['batch_size']
        # for sess
        proposal_box_list = [p[0].to(img_feats[0].device) for p in proposal_list]
        proposal_label_list = [p[2].to(img_feats[0].device) for p in proposal_list]
        proposal_score_list = [p[1].to(img_feats[0].device) for p in proposal_list]
        proposal_clspreds_list = [p[3].to(img_feats[0].device) for p in proposal_list]
        ##use threshold
        thr = 0.3  # for mean teacher thr=0.0 speed is too slow
        for idx in range(bs):
            mask = proposal_score_list[idx] > thr
            proposal_box_list[idx] = proposal_box_list[idx][mask]
            proposal_label_list[idx] = proposal_label_list[idx][mask]
            proposal_clspreds_list[idx] = proposal_clspreds_list[idx][mask]
            avg_score = proposal_score_list[idx][mask].mean()
        teacher_info['gt_bboxes_3d'] = proposal_box_list
        teacher_info['gt_labels_3d'] = proposal_label_list
        teacher_info['cls_preds'] = proposal_clspreds_list
        teacher_info['depth'] = depth
        teacher_info['avg_score'] = avg_score
        ## for distill
        teacher_info['response'] = outs[0]
        teacher_info['features'] = img_feats
        return teacher_info

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def Responseloss(
        resp_s, resp_t, gt_boxes, pc_rang, voxel_size, out_size_scale
):
    cls_s = []
    cls_t = []
    reg_t = []
    reg_s = []
    criterion = nn.L1Loss(reduce=False)
    for task_id, task_out in enumerate(resp_s):
        '''
            heatmap shape torch.Size([1, 10, 128, 128])
            reg     shape torch.Size([1, 2, 128, 128]) 
            height  shape torch.Size([1, 1, 128, 128])
            dim     shape torch.Size([1, 3, 128, 128])
            rot     shape torch.Size([1, 2, 128, 128])
            vel     shape torch.Size([1, 2, 128, 128])
        '''
        cls_s.append(_sigmoid(task_out['heatmap']))
        cls_t.append(_sigmoid(resp_t[task_id]['heatmap']))
        reg_s.append(
            torch.cat(
                [
                    task_out['reg'],
                    task_out['height'],
                    task_out['dim'],
                    task_out['rot'],
                    task_out['vel'],
                ],
                dim=1
            )
        )
        reg_t.append(
            torch.cat(
                [
                    resp_t[task_id]['reg'],
                    resp_t[task_id]['height'],
                    resp_t[task_id]['dim'],
                    resp_t[task_id]['rot'],
                    resp_t[task_id]['vel'],
                ],
                dim=1
            )
        )
    cls_s = torch.cat(cls_s, dim=1)
    cls_t = torch.cat(cls_t, dim=1)
    reg_s = torch.cat(reg_s, dim=1)
    reg_t = torch.cat(reg_t, dim=1)
    gaussian_mask = calculate_box_mask_gaussian(
        reg_s.shape,
        gt_boxes.cpu().detach().numpy(),
        pc_rang,
        voxel_size,
        out_size_scale,
    )
    cls_s_max, _ = torch.max(cls_s, dim=1)
    cls_t_max, _ = torch.max(cls_t, dim=1)
    diff_cls = criterion(cls_s_max, cls_t_max)
    diff_reg = criterion(reg_s, reg_t)
    diff_reg = torch.mean(diff_reg, dim=1)
    diff_reg = diff_reg * gaussian_mask
    diff_cls = diff_cls * gaussian_mask
    weight = gaussian_mask.sum()
    weight = reduce_mean(weight)
    loss_reg_distill = torch.sum(diff_reg) / (weight + 1e-4)
    loss_cls_distill = torch.sum(diff_cls) / (weight + 1e-4)
    return loss_cls_distill, loss_reg_distill, cls_s_max, gaussian_mask


def FeatureLoss(f_s, f_t, top_mask, gaussian_mask, ratio=0.01):
    ## gaussian_mask.shape (b,h,w)
    ## top_mask
    h, w = f_s[0].shape[-2:]
    criterion = nn.MSELoss(reduce=False)
    # breakpoint()
    with torch.no_grad():
        top2bottom = top_mask.view(-1)
        count_num = int(top2bottom.size(0) * ratio)
        sorted_vals, sorted_inds = torch.topk(top2bottom, top2bottom.size(0))
        thr = top2bottom[sorted_inds[count_num]]
    mask = top_mask > thr
    weight = mask.sum()
    weight = reduce_mean(weight)
    stu_f = f_s[0].permute(0, 2, 3, 1).contiguous()
    tea_f = f_t[0].permute(0, 2, 3, 1).contiguous()
    c = stu_f.shape[-1]
    # import pdb; pdb.set_trace()

    # todo gaussian-guided(gt_bbox) BEV distillation
    # gaussian_mask_= torch.unsqueeze(gaussian_mask, -1)
    # gaussian_mask_ = gaussian_mask_.repeat(1, 1, 1, c)
    # loss_feature = criterion(stu_f*gaussian_mask_, tea_f*gaussian_mask_)
    # loss_feature = torch.mean(loss_feature, 1)
    # loss_feature = torch.sum(loss_feature)/(weight+1e-4)

    # v2-version
    loss_feature = criterion(stu_f[mask], tea_f[mask])
    loss_feature = torch.mean(loss_feature, 1)
    loss_feature = torch.sum(loss_feature) / (weight + 1e-4)

    return loss_feature

def images_feature_loss(gt_boxes, img_imputs, img_metas, bda_params, student_image_feature, teacher_image_feature):
    '''
        get multi-views feats based on bboxes
    '''
    from ssbev.utils.coordinate_trans import lidar2img, get_lidar2global, \
        check_point_in_img, lidar3d_show, bbox3d_show
    from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB
    criterion = nn.MSELoss(reduce=False)

    scores_3d = gt_boxes[0][:, 7]
    labels_3d = gt_boxes[0][:, 8]

    imgs, sensor2keyegos, ego2globals, camera2lidars, intrins, post_rots, post_trans, bda = img_imputs  # DONE: 读取bev增强参数

    rotate_bda, scale_bda, flip_dx, flip_dy = bda_params

    # import pdb; pdb.set_trace()
    # todo image view
    imgs_draw = []
    bboxes_3d = gt_boxes[0][:, :7]

    # DONE: 针对bboxes进行bev增强的逆变换
    rotate_angle = torch.tensor(rotate_bda[0] / 180 * np.pi)
    rot_sin = torch.sin(-rotate_angle)
    rot_cos = torch.cos(-rotate_angle)
    rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
    scale_mat = torch.Tensor([[1 / scale_bda[0], 0, 0], [0, 1 / scale_bda[0], 0], [0, 0, 1 / scale_bda[0]]])
    flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    inv_rot_mat = rot_mat @ (scale_mat @ flip_mat)
    inv_rot_mat = inv_rot_mat.cuda()
    # rot_mat = bda[0]
    if flip_dy[0]:
        bboxes_3d[:, 6] = -bboxes_3d[:, 6]
    if flip_dx[0]:
        bboxes_3d[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - bboxes_3d[:, 6]

    # Compute the inverse rotation matrix
    # inv_rot_mat = rot_mat.transpose(0, 1)
    # Apply the inverse rotation
    bboxes_3d[:, :3] = (inv_rot_mat @ bboxes_3d[:, :3].unsqueeze(-1)).squeeze(-1)
    bboxes_3d[:, 3:6] /= scale_bda[0]
    bboxes_3d[:, 6] -= rotate_bda[0]

    bboxes_3d = bboxes_3d.cpu().detach().numpy()

    boxes = LB(bboxes_3d, origin=(0.5, 0.5, 0.0))
    corners_lidar = boxes.corners.numpy().reshape(-1, 3)

    imgs_path = img_metas[0]['filename']
    assert len(imgs[0]) == len(imgs_path)

    # TODO:image特征监督计算；
    # 原图尺寸大小
    image_loss = torch.zeros(1).cuda()
    for i in range(len(imgs_path)):
        file_name = imgs_path[i]
        img = cv2.imread(file_name)
        # DONE: 图像的尺寸，box的中心坐标和长宽，因子都设为1；
        boxes_img_h, boxes_img_w, boxes_centers = calc_boxes_imghwc(boxes.corners.numpy(), bboxes_3d[:, :3],
                                                                    sensor2keyegos[0][i], intrins[0][i], img.shape[0],
                                                                    img.shape[1])
        # boxes_img_h (N,), boxes_img_w (N,), boxes_centers (N, 1, 2)
        # 如果非空
        if boxes_img_h.shape[0] > 0:
            # 拼接成boxes [x,y,0,h,w](centers是二维的，需要补个0)
            boxes_img = np.concatenate(
                [boxes_centers[:, 0, :], np.zeros((boxes_centers.shape[0], 1)), boxes_img_h.reshape(-1, 1),
                 boxes_img_w.reshape(-1, 1)], axis=1)
            boxes_img = np.expand_dims(boxes_img, axis=0)
            gaussian_mask_image = calculate_box_mask_gaussian(
                [1, 1, img.shape[0], img.shape[1]],
                boxes_img,
                [0, 0, 0, img.shape[0], img.shape[1], 0],
                [1, 1, 1],
                1,
            )
            # 对mask图做图像同步的数据增强
            gaussian_mask_image = gaussian_mask_image[0].cpu().detach().numpy()  # (1, H, W)
            # 根据post_rot和post_trans对mask图进行数据增强
            # post_rot # (1, 3, 3),  post_trans # (1, 3)
            transformation_matrix = np.zeros((2, 3))
            transformation_matrix[:2, :2] = post_rots[0][i][:2, :2].cpu().detach().numpy()
            transformation_matrix[:, 2] = post_trans[0][i][:2].cpu().detach().numpy()
            aug_gaussian_mask_image = cv2.warpAffine(gaussian_mask_image, transformation_matrix, (
            img_metas[0]['batch_input_shape'][1], img_metas[0]['batch_input_shape'][1]))

            # 然后resize到特征图的大小
            target_size = (student_image_feature[0].shape[-2], student_image_feature[0].shape[-1])
            resize_aug_gaussian_mask_image = cv2.resize(aug_gaussian_mask_image, target_size)

            c = student_image_feature[0].shape[1]
            weight = resize_aug_gaussian_mask_image.sum()
            gaussian_mask_ = torch.from_numpy(resize_aug_gaussian_mask_image).transpose(0, 1).cuda()
            loss_feature = criterion(student_image_feature[0][0][i] * gaussian_mask_,
                                     teacher_image_feature[0][0][i] * gaussian_mask_)
            loss_feature = torch.mean(loss_feature, 0)
            loss_feature = torch.sum(loss_feature) / (weight + 1e-4)
            image_loss += loss_feature

    return image_loss


def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


def QFLv2(pred_sigmoid,
          teacher_sigmoid,
          weight=None,
          beta=2.0,
          reduction='mean'):
    # all goes to 0
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape)
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    pos = weight > 0

    # positive goes to bbox quality
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss


def calc_boxes_imghwc(corners_boxes, centers_boxes, camera2lidar, camera2img, img_h, img_w):
    """args
        corners_boxes: (N,8,3)
        centers_boxes: (N,1,3)
        camera2lidar: (4,4)
        camera2img: (4,4)
       return: boxes_img_h, boxes_img_w (N,)
    """
    boxes_img_h = []
    boxes_img_w = []
    boxes_centers = []
    # for corners_box in corners_boxes:
    for i in range(corners_boxes.shape[0]):
        box_img_h, box_img_w, box_center = calc_hwc(corners_boxes[i], centers_boxes[i], camera2lidar, camera2img)
        if box_img_h < 0 or box_img_w < 0 or box_img_h > img_h or box_img_w > img_w or box_center[0][0] < 0 or \
                box_center[0][0] > img_w or box_center[0][1] < 0 or box_center[0][1] > img_h:
            continue
        boxes_img_h.append(box_img_h)
        boxes_img_w.append(box_img_w)
        boxes_centers.append(box_center)
    boxes_img_h = np.array(boxes_img_h)  # (N,)
    boxes_img_w = np.array(boxes_img_w)  # (N,)
    boxes_centers = np.array(boxes_centers)  # (N, 2)

    return boxes_img_h, boxes_img_w, boxes_centers


def calc_hwc(corners_box, centers_box, camera2lidar, camera2img):
    """args
        corners_box: (8,3)
        centers_box: (3,)
        camera2lidar: (4,4)
        camera2img: (4,4)
    """
    points_lidar_homogeneous_corners = \
        np.concatenate([corners_box,
                        np.ones((corners_box.shape[0], 1),
                                dtype=corners_box.dtype)], axis=1)
    # centers_box (3,) -> (1,3)
    centers_box = np.expand_dims(centers_box, axis=0)
    points_lidar_homogeneous_centers = \
        np.concatenate([centers_box,
                        np.ones((centers_box.shape[0], 1),
                                dtype=centers_box.dtype)], axis=1)

    if isinstance(camera2lidar, torch.Tensor):
        camera2lidar = camera2lidar.cpu().numpy()
    lidar2camera = np.linalg.inv(camera2lidar)
    # import pdb; pdb.set_trace()
    points_camera_homogeneous_corners = points_lidar_homogeneous_corners @ lidar2camera.T
    points_camera_corners = points_camera_homogeneous_corners[:, :3]

    points_camera_homogeneous_centers = points_lidar_homogeneous_centers @ lidar2camera.T
    points_camera_centers = points_camera_homogeneous_centers[:, :3]

    points_camera_corners = points_camera_corners / points_camera_corners[:, 2:3]
    points_camera_centers = points_camera_centers / points_camera_centers[:, 2:3]
    if isinstance(camera2img, torch.Tensor):
        camera2img = camera2img.cpu().numpy()
    points_img_corners = points_camera_corners @ camera2img.T
    points_img_corners = points_img_corners[:, :2]

    points_img_centers = points_camera_centers @ camera2img.T
    points_img_centers = points_img_centers[:, :2]  # (N, 2)

    # 计算points_img的长宽0和1维度的最小最大值的差值，分别作为box的长宽；
    box_img_w = np.max(points_img_corners[:, 0]) - np.min(points_img_corners[:, 0])
    box_img_h = np.max(points_img_corners[:, 1]) - np.min(points_img_corners[:, 1])

    return box_img_h, box_img_w, points_img_centers


def gmm_policy(scores, given_gt_thr=0.5, policy='high'):
    """The policy of choosing pseudo label.

    The previous GMM-B policy is used as default.
    1. Use the predicted bbox to fit a GMM with 2 center.
    2. Find the predicted bbox belonging to the positive
        cluster with highest GMM probability.
    3. Take the class score of the finded bbox as gt_thr.

    Args:
        scores (nd.array): The scores.

    Returns:
        float: Found gt_thr.

    """
    if len(scores) < 4:
        return given_gt_thr
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if len(scores.shape) == 1:
        scores = scores[:, np.newaxis]
    means_init = [[np.min(scores)], [np.max(scores)]]
    weights_init = [1 / 2, 1 / 2]
    precisions_init = [[[1.0]], [[1.0]]]
    gmm = skm.GaussianMixture(
        2,
        weights_init=weights_init,
        means_init=means_init,
        precisions_init=precisions_init)
    gmm.fit(scores)
    gmm_assignment = gmm.predict(scores)
    gmm_scores = gmm.score_samples(scores)
    assert policy in ['middle', 'high']
    if policy == 'high':
        if (gmm_assignment == 1).any():
            gmm_scores[gmm_assignment == 0] = -np.inf
            indx = np.argmax(gmm_scores, axis=0)
            pos_indx = (gmm_assignment == 1) & (
                    scores >= scores[indx]).squeeze()
            pos_thr = float(scores[pos_indx].min())
            # pos_thr = max(given_gt_thr, pos_thr)
        else:
            pos_thr = given_gt_thr
    elif policy == 'middle':
        if (gmm_assignment == 1).any():
            pos_thr = float(scores[gmm_assignment == 1].min())
            # pos_thr = max(given_gt_thr, pos_thr)
        else:
            pos_thr = given_gt_thr

    return pos_thr
