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
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
#from .semi_detector import semi_detector
# from mmdet3d.models.detectors import semi_detector
from ssbev.models.detectors import semi_detector
from ..utils.gaussian_sampler import center_to_corner_box2d,calculate_box_mask_gaussian
from ..utils.torch_dist import reduce_mean

_POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
_VOXEL_SIZE = [0.1, 0.1, 0.2]
_OUT_SIZE_FACTOR = 8

@DETECTORS.register_module()
class semi_bevdet_gs_weight(semi_detector):
    def __init__(self, model, train_cfg=None, test_cfg=None):
        super(semi_bevdet_gs_weight, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.iou03, self.iou05, self.num = 0.0, 0.0, 0.0
        self.dis = [0.,0.,0.,0.]
        self.disnum=0.0
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
        super(semi_bevdet_gs_weight, self).forward_train(img_inputs, img_metas, **kwargs)
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
                #format_data[tag]['bdaiso'] = dict()
                #for ind, key in enumerate(bda_name):
                #    format_data[tag]['bdaiso'][key] = [bdaiso[ind][idx]]
            else:
                format_data[tag]['img_inputs'].append(img[idx])
                format_data[tag]['img_metas'].append(img_meta)
                format_data[tag]['gt_bboxes_3d'].append(gt_bboxes_3d[idx])
                format_data[tag]['gt_labels_3d'].append(gt_labels_3d[idx])
                for ind, key in enumerate(matrix_name):
                    format_data[tag][key].append(matrix_param[ind][idx])
                #for ind, key in enumerate(bda_name):
                #    format_data[tag]['bdaiso'][key].append(bdaiso[ind][idx])
        for tag in format_data.keys():
            format_data[tag]['img_inputs'] = [torch.stack(format_data[tag]['img_inputs'], dim=0)]
            for ind, key in enumerate(matrix_name):
                format_data[tag][key] = torch.stack(format_data[tag][key], dim=0)
                format_data[tag]['img_inputs'].append(format_data[tag][key])
                format_data[tag].pop(key)
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
        loss_resp_cls, loss_resp_reg, gaussian_mask = Responseloss_with_weight(
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

        return unsup_losses
    
    def extract_student_info(self, student_data):
        student_info={}
        img_feats, _, depth, low_features = self.teacher.extract_feat_lowlevelfeatures(points=None, 
                                                    img=student_data['img_inputs'], 
                                                    img_metas=student_data['img_metas'],
                                                    get_lowlevelfeats=True)
        outs = self.student.pts_bbox_head(img_feats)
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
    
    def statistic_box_quality(self, bs, teacher_info, student_data):
        if self.iter_count%3200 ==0 :
            from mmdet.utils import get_root_logger
            logger = get_root_logger()
            fp03, fp05 = self.iou03/self.num, self.iou05/self.num
            logger.info(f"precision0.3 is {fp03}, precision0.5 is {fp05}")
            self.iou03, self.iou05, self.num= 0.,0., 0.
        thr03, thr05=0.3, 0.5
        for i in range(bs):
            pseudo_boxes = teacher_info['gt_bboxes_3d'][i]
            gt_boxes = student_data['gt_bboxes_3d'][i]
            iou = LiDARInstance3DBoxes.overlaps(gt_boxes.to(pseudo_boxes.device), pseudo_boxes)
            val, ind = iou.max(dim=1)
            mask03 = val>thr03
            mask05 = val>thr05
            self.iou03 +=mask03.sum()
            self.iou05 +=mask05.sum()
            self.num+=iou.shape[0]
    def statistic_center_quality(self, bs, teacher_info, student_data):
        if self.iter_count%3200 ==0 :
            from mmdet.utils import get_root_logger
            logger = get_root_logger()
            for i in len(self.dis):
                fp = self.dis[i]/self.disnum
                logger.info(f"d{i} is {fp}")
                self.dis[i]=0.0
        d=[0.5, 1.0, 2.0, 4.0]
        for i in range(bs):
            pseudo_boxes = teacher_info['gt_bboxes_3d'][i].tensor
            gt_boxes = student_data['gt_bboxes_3d'][i].tensor.to(pseudo_boxes.device)
            pseudo_center = pseudo_boxes[:,:3]
            gt_center = gt_boxes[:,:3]
            dist = gt_center[:, None, :] - pseudo_center[None, :, :]
            dist = (dist**2).sum(-1)
            val, _= dist.min(1)
            self.disnum +=dist.shape[0]
            for j in range(len(d)):
                mask = val<d[j]
                self.dis[j]+=mask.sum()
        return

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

def Responseloss_with_weight(
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
    #计算学生和教师在同一位置有物体的差值, 为了提高recall，只计算教师分数高于学生的部分
    cls_s_max, _ = torch.max(cls_s, dim=1)
    cls_t_max, _ = torch.max(cls_t, dim=1)
    diff_cls = criterion(cls_s_max, cls_t_max)
    with torch.no_grad():
        diff_obj = cls_t_max - cls_s_max
        #计算同一位置的xy的偏移作为权重，中心估计对于mAP的影响很大。
        diff_reg = criterion(reg_s, reg_t)
        dist = diff_reg[:,:2]
        dist = torch.sum(dist, dim=1) + 1.
    
    diff_reg = torch.mean(diff_reg, dim=1)
    diff_reg = diff_reg * gaussian_mask * dist * (diff_obj > 0)
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
    return  loss_feature


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