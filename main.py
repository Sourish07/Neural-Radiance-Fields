import os
import time

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from nerf_dataset import TinyCybertruckDataset
from nerf_functions import get_device, get_rays, render_rays
from nerf_model import TinyNerfModel
from utils import init_xavier_uniform

torch.set_float32_matmul_precision("high")

SEED = 5
LEARNING_RATE = 5e-4
NUM_EPOCHS = 10
NUM_SAMPLES_PER_RAY = 64
BATCH_SIZE = 1

output_dir = "output"
timestamp = time.strftime("%Y%m%d-%H%M%S")
checkpoint_dir = os.path.join(output_dir, timestamp)

os.makedirs(checkpoint_dir, exist_ok=True)

# Setting seed because model is sensitive to initialization
torch.manual_seed(SEED)

device = get_device()

train_data = TinyCybertruckDataset()
train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    prefetch_factor=4,
)

testimg, testpose, testfocal = TinyCybertruckDataset(split="test")[0]
testpose = testpose.to(device)

model = TinyNerfModel()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.MSELoss()

# Using Xavier initialization because model performs better (found out through trial and error)
init_xavier_uniform(model)

model = model.to(device)

near, far = train_data.get_near_far()
H, W = train_data.get_image_size()

for i in range(NUM_EPOCHS):
    print(f"Epoch {i}")
    start_time = time.time()

    for target_image, pose, focal in tqdm(train_dataloader):
        target_image = target_image.to(device)
        pose = pose.squeeze().to(device)
        focal = focal.to(device)

        rays_o, rays_d = get_rays(H, W, focal, pose, device=device)
        rgb_map = render_rays(
            model, rays_o, rays_d, near, far, NUM_SAMPLES_PER_RAY, device=device
        )

        loss = loss_fn(rgb_map, target_image.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch took {time.time() - start_time} seconds, Loss: {loss.item()}")
    with torch.no_grad():
        rays_o, rays_d = get_rays(H, W, testfocal, testpose, device=device)
        rgb = render_rays(
            model, rays_o, rays_d, near, far, NUM_SAMPLES_PER_RAY, device=device
        )
        plt.imshow(rgb.cpu().numpy())
        plt.show()

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{i}.pt"))
