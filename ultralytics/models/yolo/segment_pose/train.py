# ultralytics/models/yolo/segment_pose/train.py

from copy import copy

# Import the base trainer
from ultralytics.models.yolo.detect import DetectionTrainer

# Import our custom model and the necessary plotting/config utilities
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_results

# We will need to create this Validator class in the next step
from copy import copy
from pathlib import Path
from typing import Dict, Optional, Union

from ultralytics.models import yolo
from ultralytics.nn.tasks import SegmentPoseModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_results

class SegmentPoseTrainer(DetectionTrainer):
    """
    A class extending the DetectionTrainer for training Segment-Pose models.

    This trainer integrates functionalities for simultaneous instance segmentation and pose estimation,
    handling the specific model, dataset, and loss requirements for this combined task.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the SegmentPoseTrainer.

        Args:
            cfg (dict): Configuration dictionary.
            overrides (dict, optional): A dictionary of parameters to override the default configuration.
            _callbacks (list, optional): A list of callback functions.
        """
        if overrides is None:
            overrides = {}
        # 1. Define a unique task name for our new model type
        overrides["task"] = "segment-pose"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Builds and returns a SegmentPoseModel.

        This method merges the model loading logic from both Segmentation and Pose trainers,
        ensuring that the `kpt_shape` is correctly passed from the dataset configuration.
        """
        # 2. Instantiate our custom SegmentPoseModel
        model = SegmentPoseModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],  # Crucial: From PoseTrainer
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        """
        Sets model attributes, including the essential `kpt_shape`.
        This is a good practice inherited from PoseTrainer.
        """
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """
        Returns a custom validator for the Segment-Pose task and defines the loss names.
        """
        # 4. Define the loss names to match our v8SegmentPoseLoss output
        # The order MUST be: [box, seg, kpt_loc, kpt_vis, cls, dfl]
        self.loss_names = ("box_loss", "seg_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss")
        # 5. Return our custom validator (which we'll need to create)
        return yolo.segment_pose.SegmentPoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_metrics(self):
        """
        Plots training and validation metrics, enabling both segmentation and pose plots.
        """
        # 6. Enable both segmentation and pose plots in results.png
        # plot_results(file=self.csv, pose=True, on_plot=self.on_plot)  # save results.png

        # fixme hank
        plot_results(dir=self.save_dir / "segment", segment=True, on_plot=self.on_plot)
        plot_results(dir=self.save_dir / "pose", pose=True,  on_plot=self.on_plot)

    def get_dataset(self):
        """
        Loads the dataset and performs a crucial check for `kpt_shape`.

        This method is inherited from the PoseTrainer to ensure the dataset is
        correctly formatted for pose estimation before training begins.
        """
        # 3. Inherit the dataset validation from PoseTrainer
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(
                f"Dataset configuration file '{self.args.data}' must contain a 'kpt_shape' key. "
                "See https://docs.ultralytics.com/datasets/pose/ for details."
            )
        return data