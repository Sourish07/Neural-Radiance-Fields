import time

import torch
from tqdm import tqdm
import wandb
from uuid import uuid4
import tempfile

from nerf_dataset import TinyCybertruckDataset
from nerf_functions import get_device, get_rays, render_rays
from nerf_model import TinyNerfModel
from utils import init_xavier_uniform

torch.set_float32_matmul_precision("high")

JOB_ID = str(uuid4())
SEED = 5
LEARNING_RATE = 5e-4
NUM_EPOCHS = 10
NUM_SAMPLES_PER_RAY = 64
BATCH_SIZE = 1

run = wandb.init(
    project="nerf-cybertruck",
    config={
        "model": "tiny_nerf_model",
        "dataset": "tiny_cybertruck_dataset",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "num_epochs": NUM_EPOCHS,
        "num_samples_per_ray": NUM_SAMPLES_PER_RAY,
    },
)

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

        run.log({"loss": loss.item()})

    with torch.no_grad():
        rays_o, rays_d = get_rays(H, W, testfocal, testpose, device=device)
        rgb = render_rays(
            model, rays_o, rays_d, near, far, NUM_SAMPLES_PER_RAY, device=device
        )
        img = rgb.mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        run.log({"generated_images": [wandb.Image(i) for i in img]})

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
        file_name = tmp_file.name
        torch.save(model.state_dict(), file_name)
        model_artifact = wandb.Artifact(
            f"{run.id}_nerf_model.pt",
            type="model",
            description=f"Epoch {i + 1}",
        )
        model_artifact.add_file(file_name, skip_cache=True)
        run.log_artifact(model_artifact)

    run.log({"epoch_time": time.time() - start_time})

run.finish()
