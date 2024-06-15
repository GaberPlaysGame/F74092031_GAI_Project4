from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Simple DIP model using CNN
class SimpleDIP(nn.Module):
    def __init__(self):
        super(SimpleDIP, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, 3, padding=1)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
if __name__ == "__main__":
    # Load a sample image
    image = img_as_float(data.camera())
    noisy_image = image + 0.1 * np.random.normal(size=image.shape)

    # Prepare the data
    x = torch.tensor(noisy_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Model and optimizer
    model = SimpleDIP()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        output = model(x)
        loss = nn.functional.mse_loss(output, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            psnr_value = psnr(x.squeeze().numpy(), output.squeeze().detach().numpy(), data_range=2147483647)
            print(f'Epoch {epoch}, Loss: {loss.item()}, PSNR: {psnr_value}')

    torch.save(model.state_dict(), f"./trained_models/dip_epochs_{epochs}.pth")

    # Show the result
    output_image = output.squeeze().detach().numpy()
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('Noisy')
    plt.imshow(noisy_image, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('DIP Result')
    plt.imshow(output_image, cmap='gray')
    plt.show()