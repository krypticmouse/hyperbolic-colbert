import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer


class PoincareProjection(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()

        self.phi_dir = nn.Linear(input_dim, embed_dim)
        self.phi_norm = nn.Linear(input_dim, 1)
        self.gamma = 1e-8

        self.init_weights()

    
    def init_weights(self):
        for module in [self.phi_dir, self.phi_norm]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)


    def forward(self, x):
        phi_dir = self.phi_dir(x)
        phi_norm = self.phi_norm(x).squeeze(-1)

        # Add small epsilon to avoid division by zero
        eps = 1e-8
        v = phi_dir / (phi_dir.norm(dim=-1, keepdim=True) + eps)
        p = torch.min(torch.sigmoid(phi_norm), torch.tensor(1.0 - self.gamma).to(x.device))

        return v * p.unsqueeze(-1)
    

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        torch.save(self.phi_dir.state_dict(), f"{path}/phi_dir.pth")
        torch.save(self.phi_norm.state_dict(), f"{path}/phi_norm.pth")


    def load_pretrained(self, path):
        self.model = BertModel.from_pretrained(path)
        self.phi_dir.load_state_dict(torch.load(f"{path}/phi_dir.pth"))
        self.phi_norm.load_state_dict(torch.load(f"{path}/phi_norm.pth"))


class HalfSpacePoincareProjection(nn.Module):
    def __init__(self,) -> None:
        super().__init__()

    def forward(self, x):
        norm_squared = torch.sum(x ** 2, dim=-1, keepdim=True)

        # Compute the y-coordinate in the Poincaré half-space
        y = 2 / (1 - norm_squared)

        # Compute the x-coordinates in the Poincaré half-space
        x = 2 * x / (1 - norm_squared)

        # Concatenate x and y to form the (n+1)-dimensional embeddings
        poincare_embeddings = torch.cat((x, y), dim=-1)

        return poincare_embeddings
