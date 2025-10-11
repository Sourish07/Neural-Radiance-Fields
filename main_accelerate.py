# accelerate launch main_accelerate.py
import os
import time

import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator
from tqdm import tqdm

from nerf_dataset import TinyCybertruckDataset
from nerf_functions import get_rays, render_rays
from nerf_model import TinyNerfModel

accelerator = Accelerator()

output_dir = "output"
timestamp = time.strftime("%Y%m%d-%H%M%S")
checkpoint_dir = os.path.join(output_dir, timestamp)

os.makedirs(checkpoint_dir, exist_ok=True)

# Setting seed because model is sensitive to initialization
seed = 5
torch.manual_seed(seed)
device = accelerator.device
train_data = TinyCybertruckDataset()
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

testimg, testpose, testfocal = TinyCybertruckDataset(split="test")[0]
testpose = testpose.to(device)

plt.imshow(testimg)
plt.show()
model = TinyNerfModel()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# Using Xavier initialization because model performs better (found out through trial and error)
# https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
for m in model.modules():
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(
            m.weight, gain=torch.nn.init.calculate_gain("relu")
        )
        torch.nn.init.zeros_(m.bias)

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)
loss_fn = torch.nn.MSELoss()

NUM_EPOCHS = 10
near, far = train_data.get_near_far()
H, W = train_data.get_image_size()
N_samples = 64

main_start_time = time.time()

for i in range(NUM_EPOCHS):
    print(f"Epoch {i}")
    start_time = time.time()

    for target_image, pose, focal in tqdm(train_dataloader):
        pose = pose.squeeze()

        rays_o, rays_d = get_rays(H, W, focal, pose, device=device)
        rgb_map = render_rays(
            model, rays_o, rays_d, near, far, N_samples, device=device
        )

        loss = loss_fn(rgb_map, target_image.squeeze())

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

    print(f"Epoch took {time.time() - start_time} seconds, Loss: {loss.item()}")
    with torch.no_grad():
        rays_o, rays_d = get_rays(H, W, testfocal, testpose, device=device)
        rgb = render_rays(model, rays_o, rays_d, near, far, N_samples, device=device)
        plt.imshow(rgb.cpu().numpy())
        plt.show()

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{i}.pt"))

print(f"Total time taken: {time.time() - main_start_time} seconds")
