## modify by Jingyu Li

import torch
import mmcv
import numpy as np
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models import DETECTORS
from mmdet3d.models.detectors import BEVDet

@DETECTORS.register_module()
class semibevdet_v2(BEVDet):
    
    def extract_feat_lowlevelfeatures(self, points, img, img_metas, get_lowlevelfeats=False, **kwargs):
        """Extract features from images and points."""
        if get_lowlevelfeats:
            img_feats, low_level_features, image_feature, depth = self.extract_img_feat_list(img, img_metas, **kwargs)
            pts_feats = None
            return (img_feats, pts_feats, depth, low_level_features, image_feature)
        else:
            img_feats, depth = self.extract_img_feat(img, img_metas, **kwargs)
            pts_feats = None
            return (img_feats, pts_feats, depth)
        
    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 8
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, camera2lidars, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda, camera2lidars]

    
    def extract_img_feat_list(self, img, img_metas, **kwargs):
        """Extract features of images."""
        #x:low-level x1:high-level
        img = self.prepare_inputs(img)
        x, _ = self.image_encoder(img[0])
        '''
            x.shape torch.Size([1, 6, 256, 16, 44]) B, N, C, imH, imW
        '''
        x_bev, depth = self.img_view_transformer([x] + img[1:7])
        x1 = self.bev_encoder(x_bev)
        return [x1],[x_bev],[x], depth
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        return super(semibevdet_v2, self).forward_train(
            points,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            gt_labels,
            gt_bboxes,
            img_inputs,
            proposals,
            gt_bboxes_ignore,
            **kwargs
        )


@DETECTORS.register_module()
class semibevdepth_v2(semibevdet_v2):
        
        def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
                    img_feats, pts_feats, depth = self.extract_feat(
                    points, img=img_inputs, img_metas=img_metas, **kwargs)
                    gt_depth = kwargs['gt_depth']

                    if isinstance(gt_depth, list):
                        gt_depth = torch.stack(gt_depth, dim=0)
                    '''
                    (Pdb) gt_depth.shape
                    torch.Size([2, 6, 256, 704]) # B, C, H, W
                    (Pdb) depth.shape
                    torch.Size([12, 59, 16, 44]) #
                    '''
                    # import pdb; pdb.set_trace()
                    loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
                    losses = dict(loss_depth=loss_depth)
                    losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                                        gt_labels_3d, img_metas,
                                                        gt_bboxes_ignore)
                    losses.update(losses_pts)
                    return losses