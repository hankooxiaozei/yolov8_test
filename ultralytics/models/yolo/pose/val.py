# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, kpt_iou


class PoseValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a pose model.

    This validator is specifically designed for pose estimation tasks, handling keypoints and implementing
    specialized metrics for pose evaluation.

    Attributes:
        sigma (np.ndarray): Sigma values for OKS calculation, either OKS_SIGMA or ones divided by number of keypoints.
        kpt_shape (List[int]): Shape of the keypoints, typically [17, 3] for COCO format.
        args (dict): Arguments for the validator including task set to "pose".
        metrics (PoseMetrics): Metrics object for pose evaluation.

    Methods:
        preprocess: Preprocess batch by converting keypoints data to float and moving it to the device.
        get_desc: Return description of evaluation metrics in string format.
        init_metrics: Initialize pose estimation metrics for YOLO model.
        _prepare_batch: Prepare a batch for processing by converting keypoints to float and scaling to original
            dimensions.
        _prepare_pred: Prepare and scale keypoints in predictions for pose processing.
        _process_batch: Return correct prediction matrix by computing Intersection over Union (IoU) between
            detections and ground truth.
        plot_val_samples: Plot and save validation set samples with ground truth bounding boxes and keypoints.
        plot_predictions: Plot and save model predictions with bounding boxes and keypoints.
        save_one_txt: Save YOLO pose detections to a text file in normalized coordinates.
        pred_to_json: Convert YOLO predictions to COCO JSON format.
        eval_json: Evaluate object detection model using COCO JSON format.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseValidator
        >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml")
        >>> validator = PoseValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize a PoseValidator object for pose estimation validation.

        This validator is specifically designed for pose estimation tasks, handling keypoints and implementing
        specialized metrics for pose evaluation.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to be used for validation.
            save_dir (Path | str, optional): Directory to save results.
            args (dict, optional): Arguments for the validator including task set to "pose".
            _callbacks (list, optional): List of callback functions to be executed during validation.

        Examples:
            >>> from ultralytics.models.yolo.pose import PoseValidator
            >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml")
            >>> validator = PoseValidator(args=args)
            >>> validator()

        Notes:
            This class extends DetectionValidator with pose-specific functionality. It initializes with sigma values
            for OKS calculation and sets up PoseMetrics for evaluation. A warning is displayed when using Apple MPS
            due to a known bug with pose models.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "pose"
        self.metrics = PoseMetrics()
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess batch by converting keypoints data to float and moving it to the device."""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].to(self.device).float()
        return batch

    def get_desc(self) -> str:
        """Return description of evaluation metrics in string format."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            # box
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            # pose
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize evaluation metrics for YOLO pose validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

    def postprocess(self, preds: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Postprocess YOLO predictions to extract and reshape keypoints for pose estimation.

        This method extends the parent class postprocessing by extracting keypoints from the 'extra'
        field of predictions and reshaping them according to the keypoint shape configuration.
        The keypoints are reshaped from a flattened format to the proper dimensional structure
        (typically [N, 17, 3] for COCO pose format).

        Args:
            preds (torch.Tensor): Raw prediction tensor from the YOLO pose model containing
                bounding boxes, confidence scores, class predictions, and keypoint data.

        Returns:
            (Dict[torch.Tensor]): Dict of processed prediction dictionaries, each containing:
                - 'bboxes': Bounding box coordinates
                - 'conf': Confidence scores
                - 'cls': Class predictions
                - 'keypoints': Reshaped keypoint coordinates with shape (-1, *self.kpt_shape)

        Note:
            If no keypoints are present in a prediction (empty keypoints), that prediction
            is skipped and continues to the next one. The keypoints are extracted from the
            'extra' field which contains additional task-specific data beyond basic detection.
        """
        # 执行NMS，保留每个物体的最佳预测结果
        """
        第一步：
        {
            'bboxes': tensor([...]),  # 最终保留的边界框
            'conf': tensor([...]),    # 对应的置信度
            'cls': tensor([...]),     # 对应的类别
            'extra': tensor([...])   # 一个“附加包裹”，里面是所有关键点的数据
        }
        """
        preds = super().postprocess(preds)
        """
        第二步：
        {
            'bboxes': tensor([...]),
            'conf': tensor([...]),
            'cls': tensor([...]),
            'keypoints': tensor_with_shape_[N, 17, 3] # 'extra' 不见了，取而代之的是结构化的 'keypoints'
        }
        """
        for pred in preds:
            pred["keypoints"] = pred.pop("extra").view(-1, *self.kpt_shape)  # remove extra if exists
        return preds

    def _prepare_batch(self, si: int, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a batch for processing by converting keypoints to float and scaling to original dimensions.

        Args:
            si (int): Batch index.
            batch (Dict[str, Any]): Dictionary containing batch data with keys like 'keypoints', 'batch_idx', etc.

        Returns:
            (Dict[str, Any]): Prepared batch with keypoints scaled to original image dimensions.

        Notes:
            This method extends the parent class's _prepare_batch method by adding keypoint processing.
            Keypoints are scaled from normalized coordinates to original image dimensions.
        """
        # pbatch:prepared batch，准备好的批次，包含了第 si 张图片的所有目标检测相关的真值信息。
        pbatch = super()._prepare_batch(si, batch)
        # batch["keypoints"]: 从整个批次的真值数据中，拿出所有关键点的数据。
        # [batch["batch_idx"]: 它记录了每一条真值数据（比如每一个框、每一个关键点集）分别属于批次中的哪一张图片。
        # [batch["batch_idx"] == si]: 只有当 batch_idx 等于我们当前要处理的图片索引 si 时，对应位置才是 True
        # 从所有关键点数据中，精确地筛选出只属于第 si 张图片的关键点。
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        # 选中所有点的x坐标，并乘以图像的高度w。
        # 选中所有点的y坐标，并乘以图像的高度h。
        # 添加到pbatch字典中
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        pbatch["keypoints"] = kpts
        return pbatch

    def _process_batch(self, preds: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Return correct prediction matrix by computing Intersection over Union (IoU) between detections and ground truth.

        Args:
            preds (Dict[str, torch.Tensor]): Dictionary containing prediction data with keys 'cls' for class predictions
                and 'keypoints' for keypoint predictions.
            batch (Dict[str, Any]): Dictionary containing ground truth data with keys 'cls' for class labels,
                'bboxes' for bounding boxes, and 'keypoints' for keypoint annotations.

        Returns:
            (Dict[str, np.ndarray]): Dictionary containing the correct prediction matrix including 'tp_p' for pose
                true positives across 10 IoU levels.

        Notes:
            `0.53` scale factor used in area computation is referenced from
            https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384.
        """
        tp = super()._process_batch(preds, batch)
        # gt_cls = batch["cls"]: 获取这张图片上所有真值物体的类别。
        gt_cls = batch["cls"]
        # 没有需要检测的物体或模型没有在这张图片上检测到任何物体。在这种情况下，不可能有任何匹配成功。
        if len(gt_cls) == 0 or len(preds["cls"]) == 0:
            tp_p = np.zeros((len(preds["cls"]), self.niou), dtype=bool)
        else:
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            # 转换为 (cx, cy, w, h)
            # [:, 2:]: 从转换后的结果中，只选取 w 和 h 这两列。
            # .prod(1): 沿着第1个维度（列维度）进行w * h乘积运算，得到每个真值框的面积。
            # * 0.53: 为了对齐官方评测工具而引入的“经验常数”。
            # area: 包含了这张图片上每个真值物体用于 OKS 计算的有效面积
            area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
            # 计算OKS (Object Keypoint Similarity)
            # sigma数组：不同关节点的容忍度是不同的。
            # area：每个真值物体的有效面积，用于归一化。
            # iou：返回一个相似度矩阵
            iou = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
            # 只有类别相同的预测和真值才可能匹配
            # 对于每个真值，它会找到与它 OKS 分数最高的那个尚未被匹配的预测
            # 然后，它会检查这个 OKS 分数是否超过了10个不同的阈值
            # 返回一个布尔矩阵，形状为 [预测数量, 10]。如果矩阵中 [i, j] 位置为 True，意味着第 i 个预测姿态成功地与一个真值匹配，并且它们的 OKS 分数超过了第 j 个阈值。
            # 即生成记录了每个预测是否配对成功，并且其配对分数是否通过了10个不同严格等级的布尔成绩单
            tp_p = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        # 将刚刚计算出的姿态评估结果 tp_p，添加到父类返回的tp字典中
        # 'tp': 基于 Bounding Box IoU 的评判结果。
        # 'tp_p': 基于 Keypoint OKS 的评判结果。
        tp.update({"tp_p": tp_p})  # update tp with kpts IoU
        return tp

    def save_one_txt(self, predn: Dict[str, torch.Tensor], save_conf: bool, shape: Tuple[int, int], file: Path) -> None:
        """
        Save YOLO pose detections to a text file in normalized coordinates.

        Args:
            predn (Dict[str, torch.Tensor]): Dictionary containing predictions with keys 'bboxes', 'conf', 'cls' and 'keypoints.
            save_conf (bool): Whether to save confidence scores.
            shape (Tuple[int, int]): Shape of the original image (height, width).
            file (Path): Output file path to save detections.

        Notes:
            The output format is: class_id x_center y_center width height confidence keypoints where keypoints are
            normalized (x, y, visibility) values for each point.
        """
        from ultralytics.engine.results import Results

        """
        包含了模型对一张图片的所有归一化的预测结果
        'bboxes': 边界框 [x_center, y_center, width, height]。
        'conf': 置信度。
        'cls': 类别 ID。
        'keypoints': 关键点 [x, y, visibility]。
        这里的“归一化”意味着所有坐标（框和关键点）的值都在 0 到 1 之间。
        """
        # 创建一个和原图一样大的纯黑图片 (height, width)。
        # 传入类别名称的映射
        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            # .cat(): 转为[N, 6],[x, y, w, h, conf, cls]
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            # [N, 17, 3]
            keypoints=predn["keypoints"],
        ).save_txt(file, save_conf=save_conf) # class_id x_center y_center width height [confidence] kpt1_x kpt1_y kpt1_vis kpt2_x kpt2_y kpt2_vis

    def pred_to_json(self, predn: Dict[str, torch.Tensor], pbatch: Dict[str, Any]) -> None:
        """
        将模型的预测数据，转换成COCO评估工具能够理解和处理的标准 JSON 格式
        Convert YOLO predictions to COCO JSON format.

        This method takes prediction tensors and a filename, converts the bounding boxes from YOLO format
        to COCO format, and appends the results to the internal JSON dictionary (self.jdict).

        Args:
            predn (Dict[str, torch.Tensor]): Prediction dictionary containing 'bboxes', 'conf', 'cls',
                and 'keypoints' tensors.
            pbatch (Dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

        Notes:
            The method extracts the image ID from the filename stem (either as an integer if numeric, or as a string),
            converts bounding boxes from xyxy to xywh format, and adjusts coordinates from center to top-left corner
            before saving to the JSON dictionary.
        """
        """
        [{
            "image_id": 123,
            "category_id": 0,  // 假设 0 代表 'person'
            "bbox": [x, y, width, height], // 像素坐标
            "score": 0.95
        }]
        """
        super().pred_to_json(predn, pbatch)
        kpts = predn["kpts"]
        # flatten(1, 2)：保持第0维不变，将第1维和第2维压平。[N, 17, 3] -> [N, 51]
        for i, k in enumerate(kpts.flatten(1, 2).tolist()):
            # 在定位到的字典中，添加一个新的keypoints键值对
            self.jdict[-len(kpts) + i]["keypoints"] = k  # keypoints

    def scale_preds(self, predn: Dict[str, torch.Tensor], pbatch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        将预测还原到原图比例
        Scales predictions to the original image size.
        """
        return {
            # {'bboxes': <一个张量，包含了被正确缩放后的边界框>}
            **super().scale_preds(predn, pbatch),
            "kpts": ops.scale_coords(
                pbatch["imgsz"],
                predn["keypoints"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估出bbox和keypoints的精度指标
        Evaluate object detection model using COCO JSON format.
        """
        anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
        pred_json = self.save_dir / "predictions.json"  # predictions
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "keypoints"], suffix=["Box", "Pose"])
