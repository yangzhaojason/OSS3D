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
from ssbev.models import Responseloss, FeatureLoss
from ..utils.gaussian_sampler import center_to_corner_box2d, calculate_box_mask_gaussian
from ..utils.torch_dist import reduce_mean


_POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
_VOXEL_SIZE = [0.1, 0.1, 0.2]
_OUT_SIZE_FACTOR = 8


class semi_bevdepth(semi_detector):
    def __init__(self, model, train_cfg=None, test_cfg=None):
        super(semi_bevdepth, self).__init__(
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
        super(semi_bevdepth, self).forward_train(img_inputs, img_metas, **kwargs)
        
        ##img_inputs len:7     [b,6,3,h,w]
        gt_bboxes_3d = kwargs.get('gt_bboxes_3d')
        gt_labels_3d = kwargs.get('gt_labels_3d')
        bdaiso = kwargs.get('bdaiso')
        import pdb; pdb.set_trace()
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
        
        feature_loss = FeatureLoss(
            student_info['features'],
            teacher_info['features'],
            gt_boxes_bev_coords,
            gt_boxes_indice,
            gaussian_mask
        )
        
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
        student_info['features'] = low_features
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
        teacher_info['features'] = low_features
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