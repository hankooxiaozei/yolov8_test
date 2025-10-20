# ultralytics/models/yolo/segment_pose/validator.py

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import OKS_SIGMA, SegmentPoseMetrics, mask_iou, kpt_iou

class SegmentPoseValidator(DetectionValidator):
    """
    A class extending DetectionValidator for validating Segment-Pose models.

    This validator integrates the validation logic for both instance segmentation and pose estimation,
    handling the combined model output and computing metrics for all three tasks: detection, segmentation, and pose.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initializes the validator, setting the task and custom metrics class."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "segment-pose"

        # 1. Use our custom combined metrics class
        self.metrics = SegmentPoseMetrics(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.sigma = None
        self.kpt_shape = None

    def get_desc(self):
        """
        Returns a formatted description of evaluation metrics for all three tasks.
        """
        # 2. Create a comprehensive description header
        return ("%22s" + "%11s" * 14) % (
            "Class", "Images", "Instances",
            "Box(P", "R", "mAP50", "mAP50-95)",
            "Mask(P", "R", "mAP50", "mAP50-95)",
            "Pose(P", "R", "mAP50", "mAP50-95)")

    def init_metrics(self, model):
        """Initializes metrics, combining logic from both validators."""
        super().init_metrics(model)
        # From PoseValidator
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

        # From SegmentationValidator
        if self.args.save_json:
            self.process = ops.process_mask_native
        else:
            self.process = ops.process_mask

    def preprocess(self, batch: Dict) -> Dict:
        """Preprocesses batch by moving both masks and keypoints to the device."""
        batch = super().preprocess(batch)
        # From SegmentationValidator
        batch["masks"] = batch["masks"].to(self.device).float()
        # From PoseValidator
        batch["keypoints"] = batch["keypoints"].to(self.device).float()
        return batch

    def postprocess(self, preds: List[torch.Tensor]) -> List[Dict]:
        """
        Postprocesses model predictions to extract boxes, masks, and keypoints.
        """
        # 3. Unpack all four outputs from our custom head
        proto = preds[1][0]  # Prototypes
        p_masks = preds[1][1]  # Mask coefficients
        p_kpts = preds[1][2]  # Keypoints

        # Standard detection postprocessing on the first output
        preds = super().postprocess(preds[0])  # This gives a list of dicts with boxes, scores, cls

        # 4. Integrate mask and keypoint processing
        for i, pred in enumerate(preds):
            # --- Mask processing (from SegmentationValidator) ---
            proto_i = proto[i]
            if pred.get("extra") is not None:
                mask_coef = pred.pop("extra")  # The detection postprocess puts mask_coefs here
                pred["masks"] = self.process(proto_i, mask_coef, pred["bboxes"], shape=self.seen_prefeito.shape[2:])
            else:  # Handle case with no detections
                pred["masks"] = torch.zeros(0, *self.seen_priors.shape[2:], dtype=torch.uint8, device=proto_i.device)

            pred["keypoints"] = torch.zeros((pred['bboxes'].shape[0], *self.kpt_shape), device=pred['bboxes'].device)
        return preds

    def _prepare_batch(self, si, batch):
        """Prepares a single batch item for metric calculation."""
        pbatch = super()._prepare_batch(si, batch)

        # From SegmentationValidator
        midx = [si] if self.args.overlap_mask else batch["batch_idx"] == si
        pbatch["masks"] = batch["masks"][midx]

        # From PoseValidator
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        pbatch["keypoints"] = kpts
        return pbatch

    def _process_batch(self, preds, batch):
        """
        Computes true positives for box, mask, and pose for a single batch.
        """
        # 5. Calculate box true positives (tp) first
        tp = super()._process_batch(preds, batch)
        gt_cls, gt_bboxes = batch["cls"], batch["bboxes"]

        # --- Mask IoU and true positives (tp_m) ---
        if "masks" in preds and "masks" in batch and len(preds["masks"]) > 0:
            gt_masks = batch["masks"]
            pred_masks = preds["masks"]
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                pred_masks = F.interpolate(pred_masks[None], gt_masks.shape[1:], mode='bilinear', align_corners=False)[
                    0]
            iou_m = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
            tp_m = self.match_predictions(preds["cls"], gt_cls, iou_m).cpu().numpy()
        else:
            tp_m = np.zeros((len(preds["cls"]), self.niou), dtype=bool)

        # --- Keypoint OKS and true positives (tp_p) ---
        if "keypoints" in preds and "keypoints" in batch and len(preds["keypoints"]) > 0:
            area = ops.xyxy2xywh(gt_bboxes)[:, 2:].prod(1) * 0.53
            iou_p = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
            tp_p = self.match_predictions(preds["cls"], gt_cls, iou_p).cpu().numpy()
        else:
            tp_p = np.zeros((len(preds["cls"]), self.niou), dtype=bool)

        # 6. Update the dictionary with all three types of true positives
        tp.update({"tp_m": tp_m, "tp_p": tp_p})
        return tp

    def eval_json(self, stats):
        """Evaluates final metrics using COCO JSON for all three tasks."""
        if self.args.save_json and self.is_coco:
            anno_json = self.data["path"] / "annotations" / f"instances_{self.args.split}2017.json"
            pred_json = self.save_dir / "predictions.json"

            # Evaluate all three: box, mask (segm), and keypoints
            return self.coco_evaluate(stats, pred_json, anno_json, ["bbox", "segm", "keypoints"],
                                      suffix=["Box", "Mask", "Pose"])
        return stats
