##version 1.0 for semi-bev
## semi-loss for foreground object  with consistency losses and  depth 

import torch
import numpy as np
import torch
import torch.nn.functional as F
import copy

from mmdet.models import DETECTORS
from mmdet3d.models.builder import build_detector
# from mmdet3d.models.builder import DETECTORS, build_detector
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
#from .semi_detector import semi_detector
# from mmdet3d.models.detectors import semi_detector
from ssbev.models.detectors import semi_detector

@DETECTORS.register_module()
class semi_bevdet_sess(semi_detector):
    def __init__(self, model, train_cfg=None, test_cfg=None):
        super(semi_bevdet_sess, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        #for test
        self.freeze("teacher")
        self.iou03, self.iou05, self.num = 0.0, 0.0, 0.0
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
        super(semi_bevdet_sess, self).forward_train(img_inputs, img_metas, **kwargs)
        
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
        
        ##remove unsup_student gt
        #del format_data['unsup_student']['gt_bboxes_3d']
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
            ##directly use gt can train normally
            ##teacher_info = teacher_data
            batch_size = teacher_data['batch_size']
        #statictis quality
        iou = self.statistic_box_quality(batch_size, teacher_info, student_data)
        ## get student boxes
        student_info = self.extract_student_info(student_data)
        #depth_loss = get_depth_loss(batch_size, student_info, teacher_info)
        unsup_losses = get_consistency_loss(batch_size, student_info, teacher_info)
        #unsup_losses['depth_loss'] = depth_loss
        return unsup_losses
    
    def extract_student_info(self, student_data):
        student_info={}
        img_feats, _, depth = self.student.extract_feat(points=None, 
                                                    img=student_data['img_inputs'], 
                                                    img_metas=student_data['img_metas'])
        outs = self.student.pts_bbox_head(img_feats)
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
        return student_info

    def extract_teacher_info(self, teacher_data):
        teacher_info={}
        img_feats, _, depth = self.teacher.extract_feat(points=None, 
                                                    img=teacher_data['img_inputs'], 
                                                    img_metas=teacher_data['img_metas'])
        outs = self.teacher.pts_bbox_head(img_feats)
        proposal_list = self.teacher.pts_bbox_head.get_bboxes_semi(
            outs, teacher_data['img_metas'], rescale=False
        )
        #bbox_results = [
        #    bbox3d2result(bboxes, scores, labels)
        #    for bboxes, scores, labels in proposal_list
        #]##boxes_3d labels_3d, scores_3d
        bs = teacher_data['batch_size']
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
