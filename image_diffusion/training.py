import math
import lightning as L
import wandb
import torch
from torch.nn import functional as F
from diffusers import UNet2DModel
from torch import nn, optim
from torchvision.datasets import CIFAR10
import torchvision
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm

class ImageDiffusionModule(nn.Module):
    def __init__(self, sigma_data=0.5):
        super().__init__()
        self.sigma_data = sigma_data
        self.unet = UNet2DModel((32, 32), time_embedding_type="fourier", num_class_embeds=10)
        ...

    def forward(self, x_noisy, t_hat, y):
        r = x_noisy / torch.sqrt(t_hat**2 + self.sigma_data**2)[..., None, None, None]

        unet_out = self.unet(r, t_hat, y)
        r_update = unet_out.sample
        r_update = r_update * t_hat[..., None, None, None] # Fix: HF does division by this

        d_skip = self.sigma_data**2 / (self.sigma_data**2 + t_hat**2)
        d_scale = self.sigma_data * t_hat / torch.sqrt(self.sigma_data**2 + t_hat**2)
        d_skip = d_skip[..., None, None, None]
        d_scale = d_scale[..., None, None, None]
        
        x_out = d_skip * x_noisy + d_scale * r_update
        return x_out

class ImageDiffusionSampler(nn.Module):
    def __init__(self, noise_steps, 
                 gamma_0=0.8, gamma_min=1.0, noise_scale=1.003, step_scale=1.0):
        super().__init__()
        self.noise_steps = noise_steps
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale


    def noise_schedule(self, t, p=7):
        sigma_min=0.002
        sigma_max=80.0
        return (sigma_max ** (1/p) + t * (sigma_min**(1/p) - sigma_max**(1/p))) ** p

    def forward(self, diffusion_module, n_classes):
        device = diffusion_module.unet.device
        noise_levels = self.noise_schedule(torch.linspace(0, 1, self.noise_steps+1, device=device))
        x_shape = (n_classes, 3, 32, 32)

        x = noise_levels[0] * torch.randn(x_shape, device=device)

        for c_prev, c in zip(noise_levels[:-1], noise_levels[1:]):

            gamma = self.gamma_0 if c > self.gamma_min else 0
            t_hat = c_prev * (gamma + 1)

            noise = self.noise_scale * torch.sqrt(t_hat**2 - c_prev**2) * torch.randn(x_shape, device=device)

            x_noisy = x+noise
            x_denoised = diffusion_module(x_noisy, t_hat, torch.arange(n_classes, device=device))

            delta = (x_noisy-x_denoised)/t_hat
            dt = c - t_hat
            x = x_noisy + self.step_scale * dt * delta

        return x * 0.5 + 0.5


        
class PLImageDiffusionModule(L.LightningModule):
    def __init__(self, sigma_data=0.5, sigma_min=0.002, sigma_max=80.0, P_mean=-1.2, P_std=1.2):
        super().__init__()
        self.model = ImageDiffusionModule(sigma_data=sigma_data)
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.P_mean = P_mean - math.log(sigma_data)
        self.P_std = P_std
        self.diffusion_sampler = ImageDiffusionSampler(noise_steps=50)

    def training_step(self, batch, batch_idx):
        x, y = batch

        t_hat = self.sigma_data * torch.exp(torch.randn(x.size(0), device=x.device) * self.P_std + self.P_mean)

        x_noisy = torch.randn_like(x) * t_hat[..., None, None, None] + x
        x_denoised = self.model(x_noisy, t_hat, y)

        loss_weight = (t_hat**2 + self.sigma_data**2) / (t_hat * self.sigma_data)**2
        loss = ((x_denoised - x)**2 * loss_weight[..., None, None, None]).mean()

        per_sample_loss = (
            ((x_denoised - x) ** 2)
            .mean(dim=(1,2,3))
            * loss_weight
        )

        mse = F.mse_loss(x_denoised, x)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/t_hat_mean", t_hat.mean())
        self.log("train/t_hat_min", t_hat.min())
        self.log("train/t_hat_max", t_hat.max())
        self.log("train/t_hat_std", t_hat.std())
        self.log("train/log_t_hat_mean", torch.log(t_hat).mean())
        self.log("train/log_t_hat_std", torch.log(t_hat).std())
        self.log("train/loss_weight_max", loss_weight.max())
        self.log("train/mse", mse)
        self.log("train/loss_weight", loss_weight.mean())

        bins = torch.tensor([0.01, 0.1, 1.0, 10.0], device=x.device)
        bucket = torch.bucketize(t_hat, bins)
        for i in range(len(bins)+1):
            mask = bucket == i
            if mask.any():
                self.log(f"train/loss_bucket_{i}", per_sample_loss[mask].mean())

        return loss

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model, norm_type=2)
        self.log(f"train/grad_norms", norms['grad_2.0_norm_total'])

    def on_train_epoch_end(self):
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            sampled_images = self.diffusion_sampler(self.model, n_classes=10)
        if was_training:
            self.model.train()

        sampled_images = torch.clamp(sampled_images, 0, 1)
        grid = torchvision.utils.make_grid(sampled_images, nrow=5)
        self.logger.experiment.log({"sampled_images": [wandb.Image(grid)]})
        

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

def main():
    T = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CIFAR10('data/datasets/', download=True, transform=T)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=16)
    wandb_logger = WandbLogger(project='image-diffusion')
    trainer = L.Trainer(max_epochs=20, logger=wandb_logger)
    model = PLImageDiffusionModule()
    trainer.fit(model, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()