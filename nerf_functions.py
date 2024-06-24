import torch

def get_rays(H, W, focal, c2w, device="cuda"):
    """
    This function generates rays that pass through each pixel of the image, starting at the camera origin.

    H: int, Height of the image
    W: int, Width of the image
    focal: float, Focal length of the camera
    c2w: torch.Tensor of shape (4, 4), Camera to world matrix
    device: str, Device to use
    returns: rays_o -> (H, W, 3), rays_d -> (H, W, 3)
    """
    
    i, j = torch.meshgrid(torch.arange(0, W), torch.arange(0, H), indexing='xy')
    i, j = i.to(device), j.to(device)
    
    dirs = torch.stack([(i - W / 2)/focal, -(j - H / 2)/focal, -torch.ones_like(i)], -1).to(device)
    dirs /= torch.norm(dirs, dim=-1, keepdim=True)
    
    # broadcasting c2w to shape (H, W, 3, 3) and multiplying with dirs of shape (H, W, 3, 1)
    # Result is a tensor of shape (H, W, 3, 1), which is why we need to squeeze the last dimension
    # Direction vectors stay relative to local origin, which is why we're not translating them
    rays_d = torch.broadcast_to(c2w[:3,:3], (H, W, 3, 3)) @ dirs[..., None]
    rays_d = rays_d.squeeze()
    
    rays_o = torch.broadcast_to(c2w[:3,-1], rays_d.shape)
    return rays_o, rays_d



def render_rays(model, rays_o, rays_d, near, far, N_samples, device="cuda"):
    """
    This function renders the rays using the NeRF model & volume rendering.

    model: nn.Module, NeRF model
    rays_o: torch.Tensor of shape (H, W, 3), Origin of the rays
    rays_d: torch.Tensor of shape (H, W, 3), Direction of the rays
    near: float, Near plane
    far: float, Far plane
    N_samples: int, Number of samples to take along each ray
    device: str, Device to use
    returns: rgb_map -> (H, W, 3), Rendered image
    """
    z_vals = torch.linspace(near, far, N_samples + 1)[:-1].to(device)
    z_vals = torch.broadcast_to(z_vals, list(rays_o.shape[:-1]) + [N_samples]).clone()
    
    z_vals += torch.rand_like(z_vals) * (far - near) / N_samples
    
    # rays_o and rays_d are of shape (H, W, 3)
    # z_vals is of shape (H, W, N_samples)
    # (H, W, 1, 3) + (H, W, 1, 3) * (H, W, N_samples, 1) -> (H, W, N_samples, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]

    rgb, sigma = model(pts)

    buffer = torch.broadcast_to(torch.tensor([1e10]).to(device), z_vals[...,:1].shape)
    dists = torch.concat([z_vals[..., 1:] - z_vals[..., :-1], buffer], -1)

    alpha = 1. - torch.exp(-sigma * dists)

    # e^{a + b} = e^a * e^b
    cumprod = torch.cumprod(1 - alpha + 1e-10, -1)
    exclusive_cumprod = torch.cat([torch.broadcast_to(torch.tensor([1.]).to(device), cumprod[...,:1].shape), cumprod[...,:-1]], -1)
    weights = alpha * exclusive_cumprod

    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    return rgb_map


def get_device():
    if torch.cuda.is_available():
        print("Using device:", torch.cuda.get_device_name(0))
        return torch.device('cuda')
    elif torch.backends.mps.is_available(): 
        # CPU might be faster for some Mac users
        # device = torch.device("cpu")
        print("Using device: Apple Silicon GPU")
        return torch.device("mps")
    else:
        print("Using device: CPU")
        return torch.device('cpu')