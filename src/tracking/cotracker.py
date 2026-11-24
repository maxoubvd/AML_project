import torch
from src.tracking.base import Tracker2D


class CoTracker3(Tracker2D):
    name = "cotracker3"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = torch.hub.load(
            "facebookresearch/co-tracker",
            "cotracker3_offline"
        )

        self.model = self.model.to(self.device).eval()

    def track(self, video, queries):

        # Prepare model inputs
        video = torch.from_numpy(video).permute(0, 3, 1, 2)
        video = video.unsqueeze(0).float().to(self.device)

        queries = torch.from_numpy(queries)
        queries = queries.unsqueeze(0).float().to(self.device)

        # Run tracking
        with torch.inference_mode():
            tracks, visible = self.model(video, queries=queries)

        return (
            tracks[0].cpu().numpy(),
            visible[0].cpu().numpy() > 0.5
        )