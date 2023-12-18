import torch, torch.nn as nn
from vae import GeniusVAE, n_channels, latent_dim, Dataset
from piq import FID, ssim

device = 'cpu'
model = GeniusVAE(n_channels=n_channels, latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load('checkpoint.pth'))
print(sum(p.numel() for p in model.parameters()), 'parameters')

latents = torch.randn(1024, latent_dim)
with torch.inference_mode():
  generated = model.inference(latents)

test_dataset = Dataset(images_dir='anime', part='test')
real = torch.stack([test_dataset[i] for i in range(1024)]).to(device)


# so at this point I have the simplest possible form, two tensors of shape (1024, 3, 64, 64)

# but fuck piq developers, I am far from computing fid between them through their library :/

class DatasetForFID(torch.utils.data.Dataset):

  def __init__(self, tensor):
    super().__init__()
    self.tensor = tensor

  def __getitem__(index):
    return {'images': self.tensor[index]}


class DataLoaderForFID:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            images, labels = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            raise StopIteration

        assert (-1 <= images.min()) and (images.max() <= 1)
        images = (images + 1) / 2
        return {'images': images, 'labels': labels}

    def __len__(self):
        return len(self.dataloader)


first_dl = DataLoaderForFID(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(generated))),
second_dl = DataLoaderForFID(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(real))),

fid_metric = FID()

print('FID:', fid_metric(fid_metric.compute_feats(first_dl), fid_metric.compute_feats(second_dl)))

print('SSIM:', ssim(generated, real))
