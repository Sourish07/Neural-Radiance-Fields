# accelerate launch --num_processes=2 main_accelerate.py
import tempfile
import time
from uuid import uuid4

import torch
from accelerate import Accelerator
from tqdm import tqdm, trange

import wandb
from nerf_dataset import TinyCybertruckDataset
from nerf_functions import get_rays, render_rays
from nerf_model import TinyNerfModel
from utils import init_xavier_uniform

accelerator = Accelerator(log_with="wandb")
device = accelerator.device

ModelClass = TinyNerfModel
DatasetClass = TinyCybertruckDataset
JOB_ID = str(uuid4())
LEARNING_RATE = 5e-4
BATCH_SIZE = 1
NUM_EPOCHS = 10
NUM_SAMPLES_PER_RAY = 64
SEED = 5

# Setting seed because model is sensitive to initialization
torch.manual_seed(SEED)
torch.set_float32_matmul_precision("high")

train_data = DatasetClass()
train_dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True
)

test_img, test_pose, test_focal = DatasetClass(split="test")[:]
test_pose = test_pose.to(device)
test_focal = test_focal.to(device)
test_img = test_img.to(device)

model = ModelClass()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

accelerator.init_trackers(
    project_name="nerf-cybertruck",
    config={
        "model": f"{ModelClass.__name__}",
        "dataset": f"{DatasetClass.__name__}",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "num_epochs": NUM_EPOCHS,
        "num_samples_per_ray": NUM_SAMPLES_PER_RAY,
    },
    # init_kwargs={"wandb": {"mode": "disabled"}},
)
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)

# Using Xavier initialization because model performs better (found out through trial and error)
init_xavier_uniform(model)

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)
loss_fn = torch.nn.MSELoss()

near, far = train_data.get_near_far()
H, W = train_data.get_image_size()


main_start_time = time.time()

for i in trange(NUM_EPOCHS, disable=not accelerator.is_main_process):
    start_time = time.time()

    for target_image, pose, focal in tqdm(
        train_dataloader,
        desc=f"Epoch {i}",
        leave=False,
        disable=not accelerator.is_main_process,
    ):
        pose.to(device)
        target_image.to(device)
        focal.to(device)

        rays_o, rays_d = get_rays(H, W, focal, pose, device=device)
        rgb_map = render_rays(
            model, rays_o, rays_d, near, far, NUM_SAMPLES_PER_RAY, device=device
        )

        loss = loss_fn(rgb_map, target_image)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        accelerator.log({"loss": loss.item()})

    accelerator.log({"epoch_time": time.time() - start_time})

    with torch.inference_mode():
        rays_o, rays_d = get_rays(H, W, test_focal, test_pose, device=device)
        rgb = render_rays(
            model, rays_o, rays_d, near, far, NUM_SAMPLES_PER_RAY, device=device
        )
        img = rgb.mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()

        accelerator.log({"generated_images": [wandb.Image(i) for i in img]})

    with tempfile.TemporaryDirectory() as tmp_dir:
        accelerator.save_state(tmp_dir)

        model_artifact = wandb.Artifact(
            JOB_ID, type="training_state", description=f"Epoch {i}"
        )
        model_artifact.add_dir(tmp_dir, skip_cache=True)
        if accelerator.is_main_process:
            wandb_tracker.log_artifact(model_artifact)

accelerator.end_training()
