##version 1.0 for semi-bev
## semi-loss for foreground object  with consistency losses and  depth 

import torch
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy

from mmdet.models import DETECTORS
from mmdet3d.models.builder import build_detector
# from mmdet3d.models.builder import DETECTORS, build_detector
#from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
#from .semi_detector import semi_detector
# from mmdet3d.models.detectors import semi_detector
from ssbev.models.detectors import semi_detector
from ..utils.gaussian_sampler import center_to_corner_box2d,calculate_box_mask_gaussian
from ..utils.torch_dist import reduce_mean

_POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
_VOXEL_SIZE = [0.1, 0.1, 0.2]
_OUT_SIZE_FACTOR = 8

@DETECTORS.register_module()
class semi_bevdet_gs(semi_detector):
    def __init__(self, model, train_cfg=None, test_cfg=None):
        super(semi_bevdet_gs, self).__init__(
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
    
    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        return imgs, [sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]
    
    def forward_train(self, img_inputs=None, img_metas=None, **kwargs):
        super(semi_bevdet_gs, self).forward_train(img_inputs, img_metas, **kwargs)
        ##img_inputs len:7     [b,6,3,h,w]
        gt_bboxes_3d = kwargs.get('gt_bboxes_3d')
        gt_labels_3d = kwargs.get('gt_labels_3d')
        bdaiso = kwargs.get('bdaiso')
        img, matrix_param = self.prepare_inputs(img_inputs)
        format_data = dict()
        matrix_name=['sensor2keyegos', 'ego2globals', 'intrins',
                'post_rots', 'post_trans', 'bda']
        bda_name = ['rotate_bda','scale_bda','flip_dx', 'flip_dy']
        for idx, img_meta in enumerate(img_metas):
            tag = img_meta['tag']
            if tag not in format_data.keys():
                format_data[tag] = dict()
                format_data[tag]['img_inputs'] = [img[idx]]
                format_data[tag]['img_metas'] = [img_meta]
                format_data[tag]['gt_bboxes_3d'] = [gt_bboxes_3d[idx]]
                format_data[tag]['gt_labels_3d'] = [gt_labels_3d[idx]]
                for ind, key in enumerate(matrix_name):
                    format_data[tag][key] = [matrix_param[ind][idx]]
                if bdaiso is not None:
                    format_data[tag]['bdaiso'] = dict()
                    for ind, key in enumerate(bda_name):
                        format_data[tag]['bdaiso'][key] = [bdaiso[ind][idx]]
            else:
                format_data[tag]['img_inputs'].append(img[idx])
                format_data[tag]['img_metas'].append(img_meta)
                format_data[tag]['gt_bboxes_3d'].append(gt_bboxes_3d[idx])
                format_data[tag]['gt_labels_3d'].append(gt_labels_3d[idx])
                for ind, key in enumerate(matrix_name):
                    format_data[tag][key].append(matrix_param[ind][idx])
                if bdaiso is not None:
                    for ind, key in enumerate(bda_name):
                        format_data[tag]['bdaiso'][key].append(bdaiso[ind][idx])
        for tag in format_data.keys():
            format_data[tag]['img_inputs'] = [torch.stack(format_data[tag]['img_inputs'], dim=0)]
            for ind, key in enumerate(matrix_name):
                format_data[tag][key] = torch.stack(format_data[tag][key], dim=0)
                format_data[tag]['img_inputs'].append(format_data[tag][key])
                format_data[tag].pop(key)
            #format_data[tag]['img_inputs'].appen([format_data[tag][name] for name in matrix_name]
            #for name in matrix_name:
        unsup_bs = len(format_data['unsup_teacher']['img_metas'])
        format_data['unsup_teacher']['batch_size']=unsup_bs
        format_data['unsup_student']['batch_size']=unsup_bs 
        
        ##remove unsup_student gt
        del format_data['unsup_student']['gt_bboxes_3d']
        del format_data['unsup_student']['gt_labels_3d']       
        
        losses =dict()
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
        if self.iter_count > self.burn_in_steps:
            unsup_weight = self.get_unsup_weight(self.weight_suppress)
            unsup_losses = self.unsup_forward_train(teacher_data=format_data['unsup_teacher'], 
                                                    student_data=format_data['unsup_student'])
            for key, val in unsup_losses.items():
                if 'loss' in key:
                    if isinstance(val, list):
                        losses[f"{key}_unsup"] = [unsup_weight * x for x in val]
                    else:
                        losses[f"{key}_unsup"] = unsup_weight * val
                else:
                    losses[key] = val
        self.iter_count +=1
        
        return losses
    
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
        #test for teacher
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(teacher_data)
            batch_size = teacher_data['batch_size']
        gt_boxes = teacher_info['gt_bboxes_3d']
        gt_boxes = torch.stack([gt.tensor for gt in gt_boxes], dim=0)
        gt_boxes_indice = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1]))
        for i in range(gt_boxes.shape[0]):
            cnt = gt_boxes[i].__len__() - 1
            while cnt > 0 and gt_boxes[i][cnt].sum() == 0:
                cnt -= 1
            gt_boxes_indice[i][: cnt + 1] = 1
        gt_boxes_indice = gt_boxes_indice.bool()
        gt_boxes_bev_coords = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1], 4, 2))
        for i in range(gt_boxes.shape[0]):
            gt_boxes_tmp = gt_boxes[i]
            gt_boxes_tmp_bev = center_to_corner_box2d(
                gt_boxes_tmp[:, :2].cpu().detach().numpy(),
                gt_boxes_tmp[:, 3:5].cpu().detach().numpy(),
                gt_boxes_tmp[:, 6].cpu().detach().numpy(),
                origin=(0.5, 0.5)
            )
            gt_boxes_bev_coords[i] = gt_boxes_tmp_bev
        gt_boxes_bev_coords = gt_boxes_bev_coords.cuda()
        gt_boxes_indice = gt_boxes_indice.cuda()
        gt_boxes_bev_coords[:, :, :, 0] = (
            gt_boxes_bev_coords[:, :, :, 0] - _POINT_CLOUD_RANGE[0]
        )/(_VOXEL_SIZE[0] * _OUT_SIZE_FACTOR)
        gt_boxes_bev_coords[:, :, :, 1] = (
            gt_boxes_bev_coords[:, :, :, 1] - _POINT_CLOUD_RANGE[1]
        )/(_VOXEL_SIZE[1] * _OUT_SIZE_FACTOR)

        ## get student boxes
        student_info = self.extract_student_info(student_data)
        loss_resp_cls, loss_resp_reg, gaussian_mask = Responseloss(
            student_info['response'],
            teacher_info['response'],
            gt_boxes,
            _POINT_CLOUD_RANGE,
            _VOXEL_SIZE,
            _OUT_SIZE_FACTOR,
        )
        unsup_losses=dict()
        unsup_losses['loss_resp_cls'] = loss_resp_cls
        unsup_losses['loss_resp_reg'] = loss_resp_reg
        '''
        feature_loss = FeatureLoss(
            student_info['features'],
            teacher_info['features'],
            gt_boxes_bev_coords,
            gt_boxes_indice,
            gaussian_mask
        )'''
        # feature_loss = BEVDistillLoss(
        #     student_info['features'][0],
        #     teacher_info['features'][0],
        #     gt_boxes_bev_coords,
        #     gt_boxes_indice,
        # )
        # unsup_losses['features_loss'] = feature_loss
        #unsup_losses = get_consistency_loss(batch_size, student_info, teacher_info)
        #depth_loss = get_depth_loss(batch_size, student_info, teacher_info)
        #unsup_losses['depth_loss'] = depth_loss
        return unsup_losses
    
    def extract_student_info(self, student_data):
        student_info={}
        img_feats, _, depth, low_features = self.teacher.extract_feat_lowlevelfeatures(points=None, 
                                                    img=student_data['img_inputs'], 
                                                    img_metas=student_data['img_metas'],
                                                    get_lowlevelfeats=True)
        outs = self.student.pts_bbox_head(img_feats)
        '''
        proposal_list = self.student.pts_bbox_head.get_bboxes_semi(
            outs, student_data['img_metas'], rescale=False
        )
        bs = student_data['batch_size']
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
        student_info['gt_bboxes_3d'] = proposal_box_list
        student_info['gt_labels_3d'] = proposal_label_list
        student_info['cls_preds'] = proposal_clspreds_list
        student_info['depth'] = depth
        '''
        student_info['response'] = outs[0]
        student_info['features'] = img_feats
        return student_info

    def extract_teacher_info(self, teacher_data):
        teacher_info={}
        img_feats, _, depth, low_features = self.teacher.extract_feat_lowlevelfeatures(points=None, 
                                                    img=teacher_data['img_inputs'], 
                                                    img_metas=teacher_data['img_metas'],
                                                    get_lowlevelfeats=True)
        outs = self.teacher.pts_bbox_head(img_feats)
        proposal_list = self.teacher.pts_bbox_head.get_bboxes_semi(
            outs, teacher_data['img_metas'], rescale=False
        )
        bs = teacher_data['batch_size']
        #for sess
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
    
    '''
    diff_cls = QFLv2(cls_s.permute(0,2,3,1).view(-1,10).contiguous(),
                     cls_t.permute(0,2,3,1).view(-1,10).contiguous(),
                     weight=gaussian_mask.view(-1),
                     reduction="None")
    diff_cls = torch.sum(diff_cls, dim=1)
    diff_cls = diff_cls * gaussian_mask.view(-1)
    '''
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
    return loss_cls_distill, loss_reg_distill, gaussian_mask

def FeatureLoss(f_s, f_t, gt_boxes_bev_coords, gt_boxes_indices, gaussian_mask):
    ## gaussian_mask.shape (b,h,w)
    h, w = f_s[0].shape[-2:]
    criterion = nn.L1Loss(reduce=False)
    mask = gaussian_mask>0.5
    weight = mask.sum()
    weight = reduce_mean(weight)
    stu_f = f_s[0].permute(0,2,3,1).contiguous()
    tea_f = f_t[0].permute(0,2,3,1).contiguous()
    loss_feature = criterion(stu_f[mask], tea_f[mask])
    loss_feature = torch.mean(loss_feature, 1)
    loss_feature = torch.sum(loss_feature)/(weight+1e-4)
    '''
    for rel loss
    stu_sample_feature = f_s.contiguous().view(
        -1, f_s.shape[-2], f_s.shape[-1]
    )
    tea_sample_feature = f_t.contiguous().view(
        -1, f_s.shape[-2], f_t.shape[-1]
    )
    stu_sample_feature = stu_sample_feature / (
        torch.norm(stu_sample_feature, dim=-1, keepdim=True) +1e-4
    )
    tea_sample_feature = tea_sample_feature / (
        torch.norm(tea_sample_feature, dim=-1, keepdim=True) +1e-4
    )
    stu_f_rel = torch.bmm(
        stu_sample_feature,
        torch.transpose(stu_sample_feature, 1, 2),
    )
    '''

    return  loss_feature
def FeaturedistillLOss(
        f_s, f_t, gt_boxes_bev_coords, gt_boxes_indices
):
    h,w=f_s[0].shape[-2:]
    gt_boxes_bev_center = torch.mean(gt_boxes_bev_coords, dim=2).unsqueeze(2)
    gt_boxes_bev_edge_1 = torch.mean(
        gt_boxes_bev_coords[:, :, [0, 1], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_2 = torch.mean(
        gt_boxes_bev_coords[:, :, [1, 2], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_3 = torch.mean(
        gt_boxes_bev_coords[:, :, [2, 3], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_4 = torch.mean(
        gt_boxes_bev_coords[:, :, [0, 3], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_all = torch.cat(
        (
            gt_boxes_bev_coords,
            gt_boxes_bev_center,
            gt_boxes_bev_edge_1,
            gt_boxes_bev_edge_2,
            gt_boxes_bev_edge_3,
            gt_boxes_bev_edge_4,
        ),
        dim=2,
    )
    gt_boxes_bev_all[:, :, :, 0] = (gt_boxes_bev_all[:, :, :, 0] - w / 2) / (w / 2)
    gt_boxes_bev_all[:, :, :, 1] = (gt_boxes_bev_all[:, :, :, 1] - h / 2) / (h / 2)
    gt_boxes_bev_all[:, :, :, [0, 1]] = gt_boxes_bev_all[:, :, :, [1, 0]]
    feature_lidar_sample = torch.nn.functional.grid_sample(
        f_s[0], gt_boxes_bev_all
    )
    feature_lidar_sample = feature_lidar_sample.permute(0, 2, 3, 1)
    feature_fuse_sample = torch.nn.functional.grid_sample(
        f_t[0], gt_boxes_bev_all
    )
    feature_fuse_sample = feature_fuse_sample.permute(0, 2, 3, 1)
    criterion = nn.L1Loss(reduce=False)
    loss_feature_distill = criterion(
        feature_lidar_sample[gt_boxes_indices], feature_fuse_sample[gt_boxes_indices]
    )
    loss_feature_distill = torch.mean(loss_feature_distill, 2)
    loss_feature_distill = torch.mean(loss_feature_distill, 1)
    loss_feature_distill = torch.sum(loss_feature_distill)
    weight = gt_boxes_indices.float().sum()
    weight = reduce_mean(weight)
    breakpoint()
    loss_feature_distill = loss_feature_distill / (weight + 1e-4)
    return loss_feature_distill
def BEVDistillLoss(bev_lidar, bev_fuse, gt_boxes_bev_coords, gt_boxes_indices):
    h, w = bev_lidar.shape[-2:]
    gt_boxes_bev_center = torch.mean(gt_boxes_bev_coords, dim=2).unsqueeze(2)
    gt_boxes_bev_edge_1 = torch.mean(
        gt_boxes_bev_coords[:, :, [0, 1], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_2 = torch.mean(
        gt_boxes_bev_coords[:, :, [1, 2], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_3 = torch.mean(
        gt_boxes_bev_coords[:, :, [2, 3], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_4 = torch.mean(
        gt_boxes_bev_coords[:, :, [0, 3], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_all = torch.cat(
        (
            gt_boxes_bev_coords,
            gt_boxes_bev_center,
            gt_boxes_bev_edge_1,
            gt_boxes_bev_edge_2,
            gt_boxes_bev_edge_3,
            gt_boxes_bev_edge_4,
        ),
        dim=2,
    )
    gt_boxes_bev_all[:, :, :, 0] = (gt_boxes_bev_all[:, :, :, 0] - w / 2) / (w / 2)
    gt_boxes_bev_all[:, :, :, 1] = (gt_boxes_bev_all[:, :, :, 1] - h / 2) / (h / 2)
    gt_boxes_bev_all[:, :, :, [0, 1]] = gt_boxes_bev_all[:, :, :, [1, 0]]
    feature_lidar_sample = torch.nn.functional.grid_sample(bev_lidar, gt_boxes_bev_all)
    feature_lidar_sample = feature_lidar_sample.permute(0, 2, 3, 1)
    feature_fuse_sample = torch.nn.functional.grid_sample(bev_fuse, gt_boxes_bev_all)
    feature_fuse_sample = feature_fuse_sample.permute(0, 2, 3, 1)
    criterion = nn.L1Loss(reduce=False)
    weight = gt_boxes_indices.float().sum()
    weight = reduce_mean(weight)
    gt_boxes_sample_lidar_feature = feature_lidar_sample.contiguous().view(
        -1, feature_lidar_sample.shape[-2], feature_lidar_sample.shape[-1]
    )
    gt_boxes_sample_fuse_feature = feature_fuse_sample.contiguous().view(
        -1, feature_fuse_sample.shape[-2], feature_fuse_sample.shape[-1]
    )
    gt_boxes_sample_lidar_feature = gt_boxes_sample_lidar_feature / (
        torch.norm(gt_boxes_sample_lidar_feature, dim=-1, keepdim=True) + 1e-4
    )
    gt_boxes_sample_fuse_feature = gt_boxes_sample_fuse_feature / (
        torch.norm(gt_boxes_sample_fuse_feature, dim=-1, keepdim=True) + 1e-4
    )
    gt_boxes_lidar_rel = torch.bmm(
        gt_boxes_sample_lidar_feature,
        torch.transpose(gt_boxes_sample_lidar_feature, 1, 2),
    )
    gt_boxes_fuse_rel = torch.bmm(
        gt_boxes_sample_fuse_feature,
        torch.transpose(gt_boxes_sample_fuse_feature, 1, 2),
    )
    gt_boxes_lidar_rel = gt_boxes_lidar_rel.contiguous().view(
        gt_boxes_bev_coords.shape[0],
        gt_boxes_bev_coords.shape[1],
        gt_boxes_lidar_rel.shape[-2],
        gt_boxes_lidar_rel.shape[-1],
    )
    gt_boxes_fuse_rel = gt_boxes_fuse_rel.contiguous().view(
        gt_boxes_bev_coords.shape[0],
        gt_boxes_bev_coords.shape[1],
        gt_boxes_fuse_rel.shape[-2],
        gt_boxes_fuse_rel.shape[-1],
    )
    loss_rel = criterion(
        gt_boxes_lidar_rel[gt_boxes_indices], gt_boxes_fuse_rel[gt_boxes_indices]
    )
    loss_rel = torch.mean(loss_rel, 2)
    loss_rel = torch.mean(loss_rel, 1)
    loss_rel = torch.sum(loss_rel)
    loss_rel = loss_rel / (weight + 1e-4)
    return loss_rel

def get_depth_loss(bs, student_info, teacher_info):
    breakpoint()
    depth_s = student_info['depth']
    depth_t = teacher_info['depth']  # n, k, h, w  1/16
    depth_s = depth_s.permute(0,2,3,1).contiguous()
    depth_t = depth_t.permute(0,2,3,1).contiguous()  # n, h, w, k
    mask = (depth_t>1e-6) * (depth_s>1e-6)
    kl = F.kl_div(depth_s.log(), depth_t, reduction='none')[mask].sum()/(6*44*16)
    weight = teacher_info['avg_score']
    return kl*weight



def get_consistency_loss(bs, student_info, teacher_info):
    center_losses, size_losses, cls_losses = [], [], []
    batch_normalizer = 0
    for idx in range(bs):
        teacher_box = teacher_info['gt_bboxes_3d'][idx].tensor
        teacher_cls = teacher_info['gt_labels_3d'][idx].view(-1,1) # Nt,1
        teacher_cls_preds = teacher_info['cls_preds'][idx] #[n,10]
        student_box = student_info['gt_bboxes_3d'][idx].tensor.to(teacher_cls.device)
        student_cls = student_info['gt_labels_3d'][idx].view(-1,1)  #Ns, 1
        student_cls_preds = student_info['cls_preds'][idx]

        num_teacher = teacher_box.shape[0]
        num_student = student_box.shape[0]
        if num_teacher==0 or num_student==0:
            batch_normalizer += 1
            continue
        teacher_centers, teacher_size = teacher_box[:, :3], teacher_box[:, 3:6]
        student_centers, student_size = student_box[:, :3], student_box[:, 3:6]
        with torch.no_grad():
            not_same_class = (teacher_cls != student_cls.T).float()
            MAX_DISTANCE = 1000000
            dist = teacher_centers[:, None, :] - student_centers[None, :, :]
            dist = (dist**2).sum(-1) ##[Nt, Ns]
            dist += not_same_class * MAX_DISTANCE
            student_dist_of_teacher, student_index_of_teacher = dist.min(1) # [Nt]
            teacher_dist_of_student, teacher_index_of_student = dist.min(0) # [Ns]
            MATCHED_DISTANCE = 1
            matched_teacher_mask = (teacher_dist_of_student < MATCHED_DISTANCE).float().unsqueeze(-1) # [Ns, 1]
            matched_student_mask = (student_dist_of_teacher < MATCHED_DISTANCE).float().unsqueeze(-1) # [Nt, 1]
        matched_teacher_centers = teacher_centers[teacher_index_of_student] # [Ns, :]
        matched_student_centers = student_centers[student_index_of_teacher] # [Nt, :]
        matched_student_cls_preds = student_cls_preds[student_index_of_teacher]
        matched_student_size = student_size[student_index_of_teacher]

        cls_loss = F.mse_loss(matched_student_cls_preds, teacher_cls_preds, reduction='none')
        cls_loss = (cls_loss * matched_student_mask).sum() / num_teacher
        size_loss = F.mse_loss(matched_student_size, teacher_size, reduction='none')
        size_loss = (size_loss * matched_student_mask).sum() / num_teacher
        center_loss = (((student_centers - matched_teacher_centers) * matched_teacher_mask).abs().sum()
                         + ((teacher_centers - matched_student_centers) * matched_student_mask).abs().sum()) \
                         / (num_teacher+num_student)
        center_losses.append(center_loss)
        cls_losses.append(cls_loss)
        size_losses.append(size_loss)
        batch_normalizer +=1
    losses = dict()
    losses['center_loss'] = sum(center_losses)/batch_normalizer
    losses['cls_loss'] = sum(cls_losses)/batch_normalizer
    losses['size_loss'] = sum(size_losses)/batch_normalizer
    return losses

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