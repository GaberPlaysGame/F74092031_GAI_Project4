from ddpm.forward_noising import forward_diffusion_sample
from ddpm.unet import SimpleUnet
from ddpm.dataloader import load_transformed_dataset
from skimage.metrics import peak_signal_noise_ratio as psnr
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

def add_noise(image, noise_level):
    return image + noise_level * torch.randn_like(image)

dip_model = SimpleDIP()
dip_optimizer = optim.Adam(dip_model.parameters(), lr=0.005)

# Load a sample image
image = img_as_float(data.camera())
noisy_image = image + 0.1 * np.random.normal(size=image.shape)

# Prepare the data
x = torch.tensor(noisy_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

for epoch in range(300):
    noise_level = np.random.uniform(0, 0.1)
    noisy_image = add_noise(x, noise_level)
    output = dip_model(noisy_image)
    loss = nn.functional.mse_loss(output, x)
    dip_optimizer.zero_grad()
    loss.backward()
    dip_optimizer.step()
    
    # Calculate metrics
    if epoch % 25 == 0:
        psnr_value = psnr(x.squeeze().numpy(), output.squeeze().detach().numpy(), data_range=2147483647)
        print(f'Epoch {epoch}, Loss: {loss.item()}, PSNR: {psnr_value}')