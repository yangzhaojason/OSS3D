#written by jingyu Li 2023.07.05

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet3d.models.builder import DETECTORS, build_detector
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
#from mmdet.core import multi_apply

#from .semi_detector import semi_detector
from mmdet3d.models.detectors import semi_detector

@DETECTORS.register_module()
class semi_bevdet_dt(semi_detector):
    def __init__(self, model, train_cfg=None, test_cfg=None):
        super(semi_bevdet_dt, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        #for test
        self.freeze("teacher")
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = train_cfg.get("unsup_weight", 1.0)
            self.sup_weight = train_cfg.get("sup_weight", 1.0)
            self.iter_count = train_cfg.get("iter", 0)
            self.burn_in_steps = train_cfg.get("burn_in_steps", 5000)
            self.weight_suppress = train_cfg.get("weight_suppress", "linear")
            self.ratio = train_cfg.get("region_ratio", 0.01)
            self.bbox_loss_type = train_cfg.get("l1", "l1")
            if self.bbox_loss_type == "l1":
                self.bbox_loss = nn.SmoothL1Loss(reduction='none')
    
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
        super(semi_bevdet_dt, self).forward_train(img_inputs, img_metas, **kwargs)
        breakpoint()
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
        print('finsh data process')
        losses =dict()
        ## supervised weight
        #sup_losses = self.student.forward_train(img_inputs = format_data['sup']['img_inputs'],
        #                                        img_metas = format_data['sup']['img_metas'],
        #                                        gt_bboxes_3d = format_data['sup']['gt_bboxes_3d'],
        #                                        gt_labels_3d = format_data['sup']['gt_labels_3d'])
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
    
    def unsup_forward_train(self, teacher_data, student_data, ratio=0.01):
        #
        teacher_info, student_info = self.extract_info(teacher_data, student_data)
        t_cls_score, t_reg_preds = teacher_info['cls'], teacher_info['reg']
        s_cls_score, s_reg_preds = student_info['cls'], student_info['reg']

        with torch.no_grad():
            count_num = int(t_cls_score.size(0) * self.ratio)
            teacher_probs = t_cls_score.sigmoid()
            max_vals = torch.max(teacher_probs, 1)[0]
            sorted_vals, sorted_inds = torch.topk(max_vals, t_cls_score.size(0))
            mask = torch.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.
            fg_num = sorted_vals[:count_num].sum()
            bg_mask = mask > 0.
        
        loss_cls = QFLv2(
            s_cls_score.sigmoid(),
            t_cls_score.sigmoid(),
            weight = mask,
            reduction="sum",
        )/fg_num

        loss_bbox = (self.bbox_loss(
            s_reg_preds[bg_mask],
            t_reg_preds[bg_mask],
        ) * t_cls_score[bg_mask].sigmoid()).mean()
        unsup_losses = dict(
            loss_cls = loss_cls,
            loss_bbox=loss_bbox
        )
        return unsup_losses
    def extract_info(self, teacher_data, student_data):
        with torch.no_grad():
            teacher_info={}
            outs, depth = self.teacher.forward_train(get_imgfeats=True, **teacher_data)
            heatmap, pred_box = self.convert_info(outs)
            teacher_info['cls'] = heatmap
            teacher_info['reg'] = pred_box
            teacher_info['depth'] = depth

        student_info ={}
        outs, depth = self.student.forward_train(get_imgfeats=True, **student_data)
        heatmap, pred_box = self.convert_info(outs)
        student_info['cls'] = heatmap
        student_info['reg'] = pred_box
        teacher_info['depth'] = depth
        
        return teacher_info, student_info
    def convert_info(self, outs):
        heatmap = outs[0][0]['heatmap'].permute(0,2,3,1).contiguous().view(-1, 10) #classes=10
        pred_box = torch.cat(
            (
                outs[0][0]['reg'],
                outs[0][0]['height'],
                outs[0][0]['dim'],
                outs[0][0]['rot'],
                outs[0][0]['vel'],
            ),
            dim=1
        ).permute(0,2,3,1).contiguous().view(-1, 10) # 2+1+3+2+2
        return heatmap, pred_box


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

##tran teacher to student
def inverse_trans(gt_boxes, rotate_angle,scale_ratio, flip_dx, flip_dy):
    gt_boxes = gt_boxes.cpu()
    rotate_angle,scale_ratio,flip_dx, flip_dy = rotate_angle.cpu() ,scale_ratio.cpu(), flip_dx.cpu(), flip_dy.cpu()
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
    if len(gt_boxes) ==0:
        gt_boxes = torch.zeros(0,9)
    final_gt_boxes= \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.0))
    return final_gt_boxes
