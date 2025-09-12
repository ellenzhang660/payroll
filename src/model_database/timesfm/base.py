import numpy as np

import timesfm
from src.model_database import model_database
from src.model_database.base import ModelChoice


class TimesFM(ModelChoice):
    """
    Absract class for generic time series dataset
    """

    def __init__(self):
        super().__init__(model=model_database["TimesFM"])

        self.pretrained = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=32,
                horizon_len=12,
                num_layers=50,
                use_positional_embedding=False,
                context_len=32,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
        )

        # def load_data():
        #     root = f"{current_working_dir}/src/database"
        #     file_path = "TimeGAN/payroll"

        #     with open(os.path.join(root, file_path, "original_data.pkl"), "rb") as f:
        #         ori_data = pickle.load(f)
        #     ori_data = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in ori_data]

        #     with open(os.path.join(root, file_path, "generated_data.pkl"), "rb") as f:
        #         syn_data = pickle.load(f)
        #     syn_data = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in syn_data]

        #     test_data = [
        #         np.sin(np.linspace(0, 20, 100)),
        #         np.sin(np.linspace(0, 20, 200)),
        #         np.sin(np.linspace(0, 20, 400)),
        #     ]

        #     return {"ori": ori_data, "syn": syn_data, "test": test_data}

    def zero_shot(self, data: np.ndarray):
        """Zero shot for TimesFM"""
        freq = [1] * len(data)
        point_forecast, _ = self.pretrained.forecast(data, freq=freq)
        return point_forecast

    def finetune(self) -> int:
        """Returns the number of samples of the time series data, assuming that each person + variable combination is the same"""
