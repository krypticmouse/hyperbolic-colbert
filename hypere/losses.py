import torch

from hypere.similarity import hyperbolic_distance

class HyperbolicContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.8, similarity_fn=hyperbolic_distance):
        super().__init__()

        self.margin = margin
        self.similarity_fn = similarity_fn

    def forward(self, query, positive, negative):
        qp_distance = self.similarity_fn(query, positive)
        qn_distance = self.similarity_fn(query, negative)

        loss = torch.relu(qp_distance - qn_distance + self.margin)
        return torch.mean(loss)
