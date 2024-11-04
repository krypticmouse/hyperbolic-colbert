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


def euclidean_distance(u, v):
    return norm(u - v, dim=-1)


def cosine_similarity(u, v):
    return torch.nn.functional.cosine_similarity(u, v, dim=-1)


def penumbral_attention(u, v, gamma, r):
    u_d = u[..., -1]
    v_d = v[..., -1]
    
    u_norm = torch.norm(u[..., :-1] - v[..., :-1], dim=-1)
    
    cone_condition = (u_norm - torch.sqrt(r**2 - u_d**2))**2 + v_d**2 < r**2
    
    sqrt_term = torch.sqrt(r**2 - u_d**2) + torch.sqrt(r**2 - v_d**2)
    max_term = torch.max(u_d, v_d, (r**2 - sqrt_term / 2 - u_norm))
    K1 = torch.exp(-gamma * max_term**2)
    
    fraction = (u_norm**2 + u_d**2 - v_d**2) / (2 * u_norm)
    K2 = torch.exp(-gamma * torch.sqrt(fraction**2 + v_d**2))
    
    return torch.where(cone_condition, K1, K2)


def umbral_attention(u, v, gamma, r):
    # Extract the last dimension values
    u_d = u[..., -1]
    v_d = v[..., -1]
    
    # Compute the norm of the vectors excluding the last dimension
    u_norm = torch.norm(u[..., :-1] - v[..., :-1], dim=-1)
    
    # Compute the max term as per the definition
    sinh_r = torch.sinh(r)
    max_term = torch.max(
        u_d, 
        v_d, 
        u_norm / (2 * sinh_r) + (u_d + v_d) / 2
    )
    
    # Compute K(u, v) using the umbral attention formula
    K = torch.exp(-gamma * max_term)
    
    return K
