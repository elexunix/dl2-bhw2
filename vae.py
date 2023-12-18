import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
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


class ResBlock(nn.Module):  # I recall these guys are useful everywhere, since the gradient flows much easier through skip-connections, but long path allow for complexity
                            # for example, recall DLA........

  def __init__(self, in_features, out_features, inner_features):
    super().__init__()
    self.stack = nn.Sequential(
      nn.BatchNorm1d(num_features=out_features),
      nn.ReLU(),
      nn.Linear(in_features=out_features, out_features=inner_features),
      nn.BatchNorm1d(num_features=inner_features),
      nn.ReLU(),
      nn.Linear(in_features=inner_features, out_features=out_features),
    )
#    self.stack = nn.Sequential(
#      nn.Linear(in_features=in_features, out_features=inner_features),
#      nn.BatchNorm1d(num_features=inner_features),
#      nn.ReLU(),
#      nn.Linear(in_features=inner_features, out_features=out_features),
#      nn.BatchNorm1d(num_features=out_features),
#    )
    self.proj = nn.Linear(in_features=in_features, out_features=out_features) if in_features != out_features else nn.Identity()

  def forward(self, x):
    x = self.proj(x)
    return x + self.stack(x)
#    x = self.proj(x) + self.stack(x)
#    return F.relu(x)


class GeniusEncoder(nn.Module):

  def __init__(self, n_channels, latent_dim):
    super().__init__()
    self.stack = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(num_features=n_channels),
      nn.ReLU(),
      nn.Conv2d(in_channels=n_channels, out_channels=2*n_channels, kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(num_features=2*n_channels),
      nn.ReLU(),
      nn.Conv2d(in_channels=2*n_channels, out_channels=4*n_channels, kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(num_features=4*n_channels),
      nn.ReLU(),
      nn.Conv2d(in_channels=4*n_channels, out_channels=8*n_channels, kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(num_features=8*n_channels),
      nn.ReLU(),
      nn.Flatten(),
      ResBlock(in_features=8*n_channels*4*4, out_features=2*latent_dim, inner_features=4*latent_dim),
      #nn.Linear(in_features=2*latent_dim, out_features=2*latent_dim),
      nn.Unflatten(dim=-1, unflattened_size=(2, latent_dim))  # mu, log sigma
    )

  def forward(self, images):
    assert images.ndim == 4  # (B, C, H, W)
    latent = self.stack(images)
    mu, sigma = latent[:, 0], torch.exp(latent[:, 1])
    return torch.distributions.Normal(loc=mu, scale=sigma)


class GeniusDecoder(nn.Module):  # GeniusEncoder and GeniusDecoder seek to invent their own new language

  def __init__(self, n_channels, latent_dim):
    super().__init__()
    self.stack = nn.Sequential(
      ResBlock(in_features=latent_dim, out_features=8*n_channels*4*4, inner_features=4*latent_dim),
      #ResBlock(in_features=2*latent_dim, out_features=8*n_channels*4*4, inner_features=4*latent_dim),
      nn.ReLU(),
      nn.Unflatten(dim=-1, unflattened_size=(8 * n_channels, 4, 4)),
      nn.ConvTranspose2d(in_channels=8*n_channels, out_channels=4*n_channels, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(num_features=4*n_channels),
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=4*n_channels, out_channels=2*n_channels, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(num_features=2*n_channels),
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=2*n_channels, out_channels=n_channels, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(num_features=n_channels),
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=n_channels, out_channels=3, kernel_size=4, stride=2, padding=1),
      nn.Tanh()
    )

  def forward(self, latent):
    return self.stack(latent)


class GeniusVAE(nn.Module):

  def __init__(self, n_channels, latent_dim):
    super().__init__()
    self.encoder = GeniusEncoder(n_channels=n_channels, latent_dim=latent_dim)
    self.decoder = GeniusDecoder(n_channels=n_channels, latent_dim=latent_dim)

  def forward(self, images):
    latent_distributions = self.encoder(images)
    latents = latent_distributions.sample()
    images = self.decoder(latents)
    #log_likelihoods = image_distributions.log_prob(images).sum(-1)
    kl_divergencies = torch.distributions.kl_divergence(latent_distributions, torch.distributions.Normal(0, 1)).sum(-1)
    return images, kl_divergencies

  def inference(self, latents):
    return self.decoder(latents)


n_channels = 64
latent_dim = 256


if __name__ == '__main__':
  batch_size = 32
  n_epochs = 1000

  images_dir = 'imagenet64/train_64x64/train_64x64'
  #images_dir = 'anime'  # images are assumed to be there just as list of 64x64 images without any subfolders
  train_dataset = Dataset(images_dir=images_dir, part='train')
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=24, shuffle=True, pin_memory=True)
  test_dataset = Dataset(images_dir=images_dir, part='test')
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=24, shuffle=True, pin_memory=True)

  device = 'cuda' or Suck
  genius_vae = GeniusVAE(n_channels=n_channels, latent_dim=latent_dim).to(device)
  print(sum(p.numel() for p in genius_vae.parameters()), 'parameters')

  optimizer = torch.optim.Adam(genius_vae.parameters(), lr=3e-4)  # reduce to 1e-6?..
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, (1e-2) ** (1 / n_epochs))
  criterion = nn.MSELoss()  # try switching to L1

  os.makedirs('samples', exist_ok=True)

  l = 0  # even 1e-6 is too much

  demo_latents = torch.randn(16, latent_dim, device=device)

  for epoch in range(n_epochs):
    genius_vae.train()
    train_reconstruction_loss, train_kl_loss, train_total_loss = 0, 0, 0
    for images in tqdm(train_dataloader):
      images = images.to(device)
      #with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      #log_likelihoods, kl_divergencies = genius_vae(images)
      #loss = kl_divergencies.mean() - log_likelihoods.mean()
      hat_images, kl_divergencies = genius_vae(images)
      reconstruction_loss = (hat_images - images).abs().mean()
      kl_loss = kl_divergencies.mean()
      loss = reconstruction_loss + l * kl_loss
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_reconstruction_loss += reconstruction_loss.item()
      train_kl_loss += kl_loss.item()
      train_total_loss += loss.item()
    train_reconstruction_loss /= len(train_dataloader)
    train_kl_loss /= len(train_dataloader)
    train_total_loss /= len(train_dataloader)
    genius_vae.eval()
    test_reconstruction_loss, test_kl_loss, test_total_loss = 0, 0, 0

    for images in tqdm(test_dataloader):
      images = images.to(device)
      #with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      with torch.no_grad():
        hat_images, kl_divergencies = genius_vae(images)
        reconstruction_loss = ((hat_images - images) ** 2).mean()
        kl_loss = kl_divergencies.mean()
        loss = reconstruction_loss + l * kl_loss
      test_reconstruction_loss += reconstruction_loss.item()
      test_kl_loss += kl_loss.item()
      test_total_loss += loss.item()
    test_reconstruction_loss /= len(test_dataloader)
    test_kl_loss /= len(test_dataloader)
    test_total_loss /= len(test_dataloader)
    print(f'Epoch {epoch}:')
    print(f'    train   reconstruction_loss {train_reconstruction_loss:.5f}, kl_loss {train_kl_loss:.5f}, total_loss {train_total_loss:.5f},')
    print(f'    test    reconstruction_loss {test_reconstruction_loss:.5f}, kl_loss {test_kl_loss:.5f}, total_loss {test_total_loss:.5f}')

    if epoch % 1 == 0:
      with torch.no_grad():
        samples = genius_vae.inference(demo_latents)
      fig, axes = plt.subplots(figsize=(16, 16), nrows=4, ncols=4)
      for i in range(4):
        for j in range(4):
          image = np.clip(samples[i * 4 + j].permute(1, 2, 0).cpu().numpy() / np.sqrt(3), -1, +1)
          axes[i, j].imshow((image * 127 + 128).astype(np.uint8), vmin=-1, vmax=+1)
      samples_path = 'samples/' + str(epoch).zfill(4) + '.png'
      plt.savefig(samples_path)
      plt.close()
      print(f'Sample images saved as {samples_path}')
      checkpoint_path = 'checkpoint.pth'
      torch.save(genius_vae.state_dict(), checkpoint_path)
      print(f'Checkpoint saved as {checkpoint_path}')
    scheduler.step()
