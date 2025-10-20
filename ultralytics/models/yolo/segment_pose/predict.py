# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops



class SegmentPosePredictor(SegmentationPredictor):
    """
    A class extending the SegmentationPredictor for prediction based on a segment-pose model.
    This class specializes in handling models that perform detection, segmentation, and pose estimation
    simultaneously. It correctly processes bounding boxes, masks, and keypoints from the model's output.
    Methods:
        construct_result: Overrides the base method to construct a result object that includes
                          bounding boxes, masks, and keypoints.
    Examples:
        >>> from ultralytics.models.yolo.segment_pose import SegmentPosePredictor
        >>>
        >>> args = dict(model="yolo11n-segpose.pt", source="path/to/images")
        >>> predictor = SegmentPosePredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the SegmentPosePredictor.
        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): List of callback functions.
        """
        # 1. Initialize the parent class (SegmentationPredictor)
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment-pose"

        # 2. Add the Apple MPS warning from PosePredictor, as it's relevant here too.
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def construct_result(self, pred, img, orig_img, img_path, proto):
        """
        Constructs a single result object with boxes, masks, and keypoints.
        This method processes the raw prediction tensor to extract and scale bounding boxes,
        generate segmentation masks, and decode and scale keypoints.
        Args:
            pred (torch.Tensor): A tensor of predictions from NMS, with shape (N, 6 + nm + nk).
                                 Columns are [box, conf, cls, mask_coefs..., kpts...].
            img (torch.Tensor): The processed input image tensor.
            orig_img (np.ndarray): The original, unprocessed image.
            img_path (str): The path to the original image file.
            proto (torch.Tensor): The prototype masks from the model's output.
        Returns:
            (ultralytics.engine.results.Results): A Results object containing boxes, masks, and keypoints.
        """
        # 1. Get model-specific parameters
        # Number of mask coefficients. The model object must have this attribute.
        nm = self.model.nm
        # Keypoint shape (e.g., (17, 3)). The model object must have this attribute.
        kpt_shape = self.model.kpt_shape
        # 2. Process masks (logic from SegmentationPredictor.construct_result)
        if not len(pred):
            # If no detections, return an empty Results object with correct attributes
            return super().construct_result(pred, img, orig_img, img_path, proto)
        # The prediction tensor `pred` from NMS contains:
        # [x1, y1, x2, y2, conf, cls, mask_coef_1, ..., mask_coef_nm, kpt1_x, kpt1_y, kpt1_vis, ..., kptN_vis]

        # Slice the prediction tensor to separate components
        boxes = pred[:, :6]  # [x1, y1, x2, y2, conf, cls]
        mask_coefs = pred[:, 6:6 + nm]

        # Process masks using the coefficients
        if self.args.retina_masks:
            boxes_scaled = ops.scale_boxes(img.shape[2:], boxes[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, mask_coefs, boxes_scaled, orig_img.shape[:2])
        else:
            masks = ops.process_mask(proto, mask_coefs, boxes[:, :4], img.shape[2:], upsample=True)

        # 3. Process keypoints (logic from PosePredictor.construct_result)
        # The keypoints data starts after the mask coefficients
        kpt_start_idx = 6 + nm
        pred_kpts = pred[:, kpt_start_idx:].view(len(pred), *kpt_shape)

        # Scale keypoints to original image dimensions
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        # 4. Filter out instances where the mask is empty after processing
        # This is an important step from SegmentationPredictor
        if masks is not None:
            keep = masks.sum((1, 2)) > 0
            boxes, masks, pred_kpts = boxes[keep], masks[keep], pred_kpts[keep]
        # 5. Construct the final Results object
        # First, scale the bounding boxes for the final result
        boxes[:, :4] = ops.scale_boxes(img.shape[2:], boxes[:, :4], orig_img.shape)

        # Create Results object and populate all fields
        # result = super().construct_result(boxes, img, orig_img, img_path, masks)
        # result.update(keypoints=pred_kpts)
        result = Results(orig_img, path=img_path, names=self.model.names, boxes=boxes, masks=masks, keypoints=pred_kpts)
        return result