import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from tqdm import tqdm, trange

class Dataset(torch.utils.data.Dataset):

  def __init__(self, images_dir, part):
    super().__init__()
    self.filenames = [os.path.join(images_dir, filename) for filename in os.listdir(images_dir)]
    assert part in ['train', 'test']
    if part == 'train':
      self.filenames = [filename for filename in self.filenames if hash(filename) % 10 < 9]  # pseudo-randomly select 90%
    else:
      self.filenames = [filename for filename in self.filenames if hash(filename) % 10 == 9]  # these are the rest 10%
    print('dataset contains', len(self.filenames), 'images')

  def __getitem__(self, index):
    image = np.asarray(Image.open(self.filenames[index]))
    return ((torch.tensor(image).permute(2, 0, 1) / 127.5 - 1) * np.sqrt(3)).float()  # for motivation: variance of uniform distribution

  def __len__(self):
    return len(self.filenames)


class DenoisingGeniusModel(nn.Module):

  def __init__(self, n_channels):
    super().__init__()
    self.down1 = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(num_features=n_channels),
      nn.ReLU(),
    )
    self.down2 = nn.Sequential(
      nn.Conv2d(in_channels=n_channels, out_channels=2*n_channels, kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(num_features=2*n_channels),
      nn.ReLU(),
    )
    self.down3 = nn.Sequential(
      nn.Conv2d(in_channels=2*n_channels, out_channels=4*n_channels, kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(num_features=4*n_channels),
      nn.ReLU(),
    )
    self.down4 = nn.Sequential(
      nn.Conv2d(in_channels=4*n_channels, out_channels=8*n_channels, kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(num_features=8*n_channels),
      nn.ReLU(),
    )
    self.up4 = nn.Sequential(
      nn.ConvTranspose2d(in_channels=8*n_channels, out_channels=4*n_channels, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(num_features=4*n_channels),
      nn.ReLU(),
    )
    self.up3 = nn.Sequential(
      nn.ConvTranspose2d(in_channels=8*n_channels, out_channels=2*n_channels, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(num_features=2*n_channels),
      nn.ReLU(),
    )
    self.up2 = nn.Sequential(
      nn.ConvTranspose2d(in_channels=4*n_channels, out_channels=n_channels, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(num_features=n_channels),
      nn.ReLU(),
    )
    self.up1 = nn.Sequential(
      nn.ConvTranspose2d(in_channels=2*n_channels, out_channels=3, kernel_size=4, stride=2, padding=1),
    )

  def forward(self, x, t):
    a = self.down1(x)
    b = self.down2(a)
    c = self.down3(b)
    d = self.down4(c)
    x = self.up4(d)
    x = self.up3(torch.cat([x, c], 1))
    x = self.up2(torch.cat([x, b], 1))
    x = self.up1(torch.cat([x, a], 1))
    return x


batch_size = 32
n_channels = 256
n_epochs = 100
n_steps = 100

images_dir = 'imagenet64/train_64x64/train_64x64'
train_dataset = Dataset(images_dir=images_dir, part='train')  # images are assumed to be in './data'
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=24, shuffle=True, pin_memory=True)
test_dataset = Dataset(images_dir=images_dir, part='test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=24, shuffle=True, pin_memory=True)

device = 'cuda'
denoising_genius = DenoisingGeniusModel(n_channels=n_channels).to(device)
print(sum(p.numel() for p in denoising_genius.parameters()), 'parameters')

optimizer = torch.optim.Adam(denoising_genius.parameters(), lr=1e-3)  # reduce to 1e-6?..
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, (1e-2) ** (1 / n_epochs))
criterion = nn.MSELoss()  # try switching to L1

#betas = np.exp(np.linspace(-10, 0, 100))  # my own variance shedule, gradually increases from very small, sums up to 10.4; this sucks...
betas = 30 * np.linspace(0, 0.1, 100) ** 2  # my own variance schedule, gradually increases from small, sums up to 10.1
assert sum(betas) > 10
alphas = torch.tensor(np.cumprod(1 - betas), dtype=torch.float, device=device)  # frac. image variance left

os.makedirs('samples', exist_ok=True)
for epoch in range(n_epochs):
  denoising_genius.train()
  train_loss = 0
  for images in tqdm(train_dataloader):
    images = images.to(device)
    noises = torch.randn_like(images)
    ts = torch.randint(n_steps, size=(len(images),), device=device)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      noisy_images = torch.sqrt(alphas[ts]).view(-1, 1, 1, 1) * images + torch.sqrt(1 - alphas[ts]).view(-1, 1, 1, 1) * noises
      loss = criterion(denoising_genius(noisy_images, ts), noises)
    train_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  train_loss /= len(train_dataloader)
  denoising_genius.eval()
  test_loss = 0
  for images in tqdm(test_dataloader):
    images = images.to(device)
    noises = torch.randn_like(images)
    ts = torch.randint(n_steps, size=(len(images),), device=device)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      noisy_images = torch.sqrt(alphas[ts]).view(-1, 1, 1, 1) * images + torch.sqrt(1 - alphas[ts]).view(-1, 1, 1, 1) * noises
      loss = criterion(denoising_genius(noisy_images, ts), noises)
    test_loss += loss.item()
  test_loss /= len(test_dataloader)
  print(f'Epoch {epoch}: train loss {train_loss}, test loss {test_loss}')
  if epoch % 1 == 0:
    samples = torch.randn(16, 3, 64, 64, device=device)
    with torch.no_grad():
      for t in range(n_steps - 1, 0, -1):
        ts = torch.tensor(t, device=device).view(-1, 1, 1, 1)
        beta = betas[torch.tensor(t, device=device)]
        predicted_clean = (samples - torch.sqrt(1 - alphas[ts]) * denoising_genius(samples, ts)) / torch.sqrt(alphas[ts])
        noises = torch.randn_like(samples)
        samples = torch.sqrt(alphas[ts - 1]) * predicted_clean + torch.sqrt(1 - alphas[ts - 1]) * noises
    fig, axes = plt.subplots(figsize=(16, 16), nrows=4, ncols=4)
    for i in range(4):
      for j in range(4):
        image = np.clip(samples[i * 4 + j].permute(1, 2, 0).cpu().numpy() / np.sqrt(3), -1, +1)
        axes[i, j].imshow((image * 127 + 128).astype(np.uint8), vmin=-1, vmax=+1)
    samples_path = 'samples/' + str(epoch).zfill(4) + '.png'
    plt.savefig(samples_path)
    plt.close()
    print(f'Sample images saved as {samples_path}')
  scheduler.step()
