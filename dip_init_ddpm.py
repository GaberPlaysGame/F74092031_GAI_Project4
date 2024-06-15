
from ddpm.forward_noising import forward_diffusion_sample
from ddpm.unet import SimpleUnet
from ddpm.dataloader import load_transformed_dataset
from dip import SimpleDIP
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import torch.optim as optim
import logging
from tqdm import tqdm
from skimage import data, img_as_float
import numpy as np

logging.basicConfig(level=logging.INFO)


def get_loss(model, x_0, t, device):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    # return F.l1_loss(noise, noise_pred)
    return F.mse_loss(noise, noise_pred)

# Load a sample image
image = img_as_float(data.camera())
noisy_image = image + 0.1 * np.random.normal(size=image.shape)

# Prepare the data
x = torch.tensor(noisy_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Assume DIP model has been trained and we have the DIP output
    dip_model = SimpleDIP()
    dip_optimizer = optim.Adam(dip_model.parameters(), lr=0.01)

    # Train DIP model
    for epoch in range(100):
        dip_output = dip_model(x)
        dip_loss = nn.functional.mse_loss(dip_output, x)
        dip_optimizer.zero_grad()
        dip_loss.backward()
        dip_optimizer.step()
    
    BATCH_SIZE = 128
    dataloader = load_transformed_dataset(batch_size=BATCH_SIZE)

    ddpm_model  = SimpleUnet()
    T = 300
    epochs = 10

    ddpm_model .to(device)
    ddpm_optimizer  = Adam(ddpm_model .parameters(), lr=0.001)

    initial_prior = dip_output.detach()

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for batch_idx, (batch, _) in enumerate(dataloader):
            ddpm_optimizer.zero_grad()

            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            # x_t = initial_prior * torch.sqrt(torch.tensor(ddpm_model.alpha_cumprod[t])) + torch.sqrt(torch.tensor(1 - ddpm_model.alpha_cumprod[t])) * torch.randn_like(initial_prior)
            x_t, _ = forward_diffusion_sample(initial_prior, t, device)
            loss = get_loss(ddpm_model , x_t, t, device=device)
            ddpm_optimizer.zero_grad()
            loss.backward()
            ddpm_optimizer.step()

            if batch_idx % 50 == 0:
                logging.info(f"Epoch {epoch} | Batch index {batch_idx:03d} Loss: {loss.item()}")

            pbar.set_postfix(loss=loss.item())

        pbar.close()

    torch.save(ddpm_model .state_dict(), f"./trained_models/dip_ddpm_epochs_{epochs}.pth")