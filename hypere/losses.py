import torch
from torch.linalg import norm

def hyperbolic_distance(u, v, max_norm=None):
    numerator = 2 * norm(u - v, dim=-1) ** 2
    denominator = (1 - norm(u, dim=-1) ** 2) * (1 - norm(v, dim=-1) ** 2)
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    denominator = torch.clamp(denominator, min=epsilon)
    
    # Ensure the argument to acosh is >= 1
    acosh_arg = torch.clamp(1 + (numerator / denominator), min=1.0 + epsilon)
    
    return torch.acosh(acosh_arg) / (max_norm or 1.0)


MAX_GAMMA_DISTANCE = hyperbolic_distance(torch.tensor([0, 1-(1e-8)]), torch.tensor([0, (1e-8)-1]))


def cosine_similarity(u, v):
    return torch.nn.functional.cosine_similarity(u, v, dim=-1)


def euclidean_distance(u, v):
    return norm(u - v, dim=-1)


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
