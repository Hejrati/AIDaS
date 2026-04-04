from oct_image_segmentation_models.common.utils import get_timestamp


class MLflowParameters:
    def __init__(
        self,
        tracking_uri: str = "mlruns",
        username: str = None,
        password: str = None,
        experiment: str = f"experiment-{get_timestamp()}",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.username = username
        self.password = password
        self.experiment = experiment
