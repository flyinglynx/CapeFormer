'''
A simple direct regression-based model for CAPE
The query tokens are directly processed by a shared MLP to generate 2D coordinates.
'''
import math

import cv2
import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.visualization.image import imshow

from mmpose.models import builder
from mmpose.models.detectors.base import BasePose
from mmpose.models.builder import POSENETS

import torch


@POSENETS.register_module()
class TransformerPoseTwoStage(BasePose):
    """Few-shot keypoint detectors.
    Args:
        encoder_sample (dict): Backbone modules to extract feature.
        encoder_query (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        share_backbone (bool): Whether to share the backbone for support and query.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (dict): Config for loss. Default: None.
    """

    def __init__(self,
                 encoder_config,
                 keypoint_head,
                 share_backbone=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()

        if not share_backbone:
            self.encoder_sample = builder.build_backbone(
                encoder_config
            )  # the encoder sample should be a dict of backbone config.
            self.encoder_query = builder.build_backbone(encoder_config)
        else:
            self.encoder_sample = self.encoder_query = builder.build_backbone(
                encoder_config)

        self.keypoint_head = builder.build_head(keypoint_head)
        self.init_weights(pretrained=pretrained)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg  # {'flip_test': False, 'post_process': 'default', 'shift_heatmap': True, 'modulate_kernel': 11}
        self.target_type = test_cfg.get('target_type',
                                        'GaussianHeatMap')  # GaussianHeatMap

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.encoder_sample.init_weights(pretrained)
        self.encoder_query.init_weights(pretrained)
        self.keypoint_head.init_weights()

    def forward(self,
                img_s,
                img_q,
                target_s=None,
                target_weight_s=None,
                target_q=None,
                target_weight_q=None,
                img_metas=None,
                return_loss=True,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C (Default: 3)
            img height: imgH
            img weight: imgW
            heatmaps height: H
            heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.

        Returns:
            dict|tuple: if `return loss` is true, then return losses.
              Otherwise, return predicted poses, boxes and image paths.
        """
        if return_loss:
            return self.forward_train(img_s, target_s, target_weight_s, img_q,
                                      target_q, target_weight_q, img_metas,
                                      **kwargs)
        else:
            return self.forward_test(img_s, target_s, target_weight_s, img_q,
                                     target_q, target_weight_q, img_metas,
                                     **kwargs)

    def forward_train(self, img_s, target_s, target_weight_s, img_q, target_q,
                      target_weight_q, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""

        feature_s = [
            self.encoder_sample(img) for img in img_s
        ]  # list of support features from different shot. Each 'img' is in batch format.
        feature_q = self.encoder_query(img_q)  # [bs, 3, h, w]

        mask_s = target_weight_s[0]  # [bs, num_query, 1]
        for target_weight in target_weight_s:
            mask_s = mask_s * target_weight

        output, initial_proposals, similarity_map = self.keypoint_head(
            feature_q, feature_s, target_s, mask_s
        )  # target_s: list of [bs, num_query, mh, mw], mask_s: [bs, num_query, 1]

        # parse the img meta to get the target keypoints
        target_keypoints = self.parse_keypoints_from_img_meta(
            img_metas, feature_q.device)  # [bs, num_query, 2]
        target_sizes = torch.tensor([
            img_q.shape[-2], img_q.shape[-1]
        ]).unsqueeze(0).repeat(img_q.shape[0], 1,
                               1)  # [bs, 2] batch_idx, width, height.

        # if return loss
        losses = dict()
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output, initial_proposals, similarity_map, target_keypoints,
                target_q, target_weight_q * mask_s, target_sizes
            )  # make sure the keypoints co-exist in query and support.
            losses.update(keypoint_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                output[-1], target_keypoints, target_weight_q * mask_s,
                target_sizes)
            losses.update(keypoint_accuracy)

        return losses

    def forward_test(self,
                     img_s,
                     target_s,
                     target_weight_s,
                     img_q,
                     target_q,
                     target_weight_q,
                     img_metas=None,
                     vis_similarity_map=False,
                     vis_offset=False,
                     **kwargs):
        """Defines the computation performed at every call when testing."""
        batch_size, _, img_height, img_width = img_q.shape

        result = {}
        feature_s = [self.encoder_sample(img) for img in img_s]
        feature_q = self.encoder_query(img_q)

        mask_s = target_weight_s[0]
        # mask_s[:,30:] = 0
        for target_weight in target_weight_s:
            mask_s = mask_s * target_weight
        output, initial_proposals, similarity_map = self.keypoint_head(
            feature_q, feature_s, target_s, mask_s)
        predicted_pose = output[-1].detach().cpu().numpy(
        )  # [bs, num_query, 2]

        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode(
                img_metas, predicted_pose, img_size=[img_width, img_height])
            result.update(keypoint_result)

        if vis_similarity_map:
            similarity_map_shape = similarity_map.shape
            similarity_map = similarity_map.reshape(*similarity_map_shape[:2],
                                                    -1)
            similarity_map = (similarity_map - torch.min(
                similarity_map, dim=2)[0].unsqueeze(2)) / (
                    torch.max(similarity_map, dim=2)[0].unsqueeze(2) -
                    torch.min(similarity_map, dim=2)[0].unsqueeze(2) + 1e-10)
            result.update({
                "similarity_map":
                similarity_map.reshape(similarity_map_shape)[0].cpu().numpy()
            })

        if vis_offset:
            result.update({
                "points":
                torch.cat((initial_proposals, output.squeeze())).cpu().numpy()
            })

        result.update({"sample_image_file": img_metas[0]['sample_image_file']})

        return result

    def parse_keypoints_from_img_meta(self, img_meta, device):
        """Parse keypoints from the img_meta.

        Args:
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Keypoints coordinates of query images.
        """
        query_kpt = torch.stack([
            torch.tensor(info['query_joints_3d']).to(device)
            for info in img_meta
        ],
                                dim=0)[:, :, :2]  # [bs, num_query, 2]
        return query_kpt

    # UNMODIFIED
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_limb_color=None,
                    radius=4,
                    text_color=(255, 0, 0),
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_limb_color (np.array[Mx3]): Color of M limbs.
                If None, do not draw limbs.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """

        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape

        bbox_result = []
        pose_result = []
        for res in result:
            bbox_result.append(res['bbox'])
            pose_result.append(res['keypoints'])

        if len(bbox_result) > 0:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            mmcv.imshow_bboxes(
                img,
                bboxes,
                colors=bbox_color,
                top_k=-1,
                thickness=thickness,
                show=False,
                win_name=win_name,
                wait_time=wait_time,
                out_file=None)

            for person_id, kpts in enumerate(pose_result):
                # draw each point on image
                if pose_kpt_color is not None:
                    assert len(pose_kpt_color) == len(kpts), (
                        len(pose_kpt_color), len(kpts))
                    for kid, kpt in enumerate(kpts):
                        x_coord, y_coord, kpt_score = int(kpt[0]), int(
                            kpt[1]), kpt[2]
                        if kpt_score > kpt_score_thr:
                            img_copy = img.copy()
                            r, g, b = pose_kpt_color[kid]
                            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                       radius, (int(r), int(g), int(b)), -1)
                            transparency = max(0, min(1, kpt_score))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

                # draw limbs
                if skeleton is not None and pose_limb_color is not None:
                    assert len(pose_limb_color) == len(skeleton)
                    for sk_id, sk in enumerate(skeleton):
                        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1,
                                                                  1]))
                        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1,
                                                                  1]))
                        if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                                and pos1[1] < img_h and pos2[0] > 0
                                and pos2[0] < img_w and pos2[1] > 0
                                and pos2[1] < img_h
                                and kpts[sk[0] - 1, 2] > kpt_score_thr
                                and kpts[sk[1] - 1, 2] > kpt_score_thr):
                            img_copy = img.copy()
                            X = (pos1[0], pos2[0])
                            Y = (pos1[1], pos2[1])
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                            angle = math.degrees(
                                math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            stickwidth = 2
                            polygon = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(stickwidth)), int(angle),
                                0, 360, 1)

                            r, g, b = pose_limb_color[sk_id]
                            cv2.fillConvexPoly(img_copy, polygon,
                                               (int(r), int(g), int(b)))
                            transparency = max(
                                0,
                                min(
                                    1, 0.5 *
                                    (kpts[sk[0] - 1, 2] + kpts[sk[1] - 1, 2])))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

        show, wait_time = 1, 1
        if show:
            height, width = img.shape[:2]
            max_ = max(height, width)

            factor = min(1, 800 / max_)
            enlarge = cv2.resize(
                img, (0, 0),
                fx=factor,
                fy=factor,
                interpolation=cv2.INTER_CUBIC)
            imshow(enlarge, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img
