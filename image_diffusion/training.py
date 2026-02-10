import argparse
import math
from attr import dataclass
import lightning as L
from matplotlib import pyplot as plt
from tqdm import tqdm
import wandb
import torch
from torch.nn import functional as F
from diffusers import UNet2DModel
from torch import nn, optim
from torchvision.datasets import CIFAR10
import torchvision
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm
import yaml

@dataclass
class Config:
    # Diffusion hyperparameters
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    P_mean: float = -1.2
    P_std: float = 1.2

    # Sampler hyperparameters
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.0
    noise_steps: int = 50

    # Training parameters
    batch_size: int = 128
    num_epochs: int = 40
    learning_rate: float = 1e-4
    learning_rate_warmup: bool = True
    learning_rate_warmup_steps: int = 500
    continue_from_checkpoint: str = None

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


    def noise_schedule(self, t, p=7, append_zero=False):
        sigma_min=0.002
        sigma_max=80.0
        schedule = (sigma_max ** (1/p) + t * (sigma_min**(1/p) - sigma_max**(1/p))) ** p
        if append_zero:
            schedule = torch.cat([schedule, torch.zeros(1, device=schedule.device)])
        return schedule

    def first_order_step(self, diffusion_module, x, y, c_prev, c):
        x_denoised = diffusion_module(x, c_prev, y)
        delta = (x-x_denoised) / c_prev
        dt = c-c_prev
        x_next = x + dt * delta
        return x_next

    def second_order_step(self, diffusion_module, x, y, c_prev, c):
        x_denoised = diffusion_module(x, c_prev, y)
        delta = (x-x_denoised) / c_prev
        dt = c - c_prev
        x_next = x + dt * delta

        if c > 0:
            x_prime_denoised = diffusion_module(x_next, c, y)
            delta_prime = (x_next - x_prime_denoised) / c
            x_next = x + 0.5 * dt * (delta + delta_prime)
        return x_next

    def stochastic_solver_step(self, diffusion_module, x, y, c_prev, c):
        gamma = self.gamma_0 if c > self.gamma_min else 0
        t_hat = c_prev * (gamma + 1)
        noise = self.noise_scale * torch.sqrt(t_hat**2 - c_prev**2) * torch.randn_like(x)
        x_noisy = x + noise
        x_next = self.first_order_step(diffusion_module, x_noisy, y, t_hat, c)
        return x_next

    def forward(self, diffusion_module, n_classes, solver='second_order'):
        device = diffusion_module.unet.device
        if solver == 'second_order': 
            noise_steps = self.noise_steps // 2
        else:
            noise_steps = self.noise_steps

        noise_levels = self.noise_schedule(torch.linspace(0, 1, noise_steps, device=device), append_zero=True)
        x_shape = (n_classes, 3, 32, 32)
        x = noise_levels[0] * torch.randn(x_shape, device=device)

        y = torch.arange(n_classes, device=device)
        for c_prev, c in tqdm(zip(noise_levels[:-1], noise_levels[1:])):
            if solver == 'first_order':
                x = self.first_order_step(diffusion_module, x, y, c_prev, c)
            elif solver == 'second_order':
                x = self.second_order_step(diffusion_module, x, y, c_prev, c)
            elif solver == 'stochastic_solver':
                x = self.stochastic_solver_step(diffusion_module, x, y, c_prev, c)
            else:
                raise ValueError(f"Unknown solver: {solver}")
            

        return x * 0.5 + 0.5


        
class PLImageDiffusionModule(L.LightningModule):
    def __init__(self, config: Config=None):
        super().__init__()
        if config is None:
            config = Config()
        self.config = config
        self.model = ImageDiffusionModule(sigma_data=config.sigma_data)
        self.sigma_data = config.sigma_data
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.P_mean = config.P_mean - math.log(config.sigma_data)
        self.P_std = config.P_std
        self.diffusion_sampler = ImageDiffusionSampler(noise_steps=config.noise_steps)

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
        optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)

        if self.config.learning_rate_warmup:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step+1)/self.config.learning_rate_warmup_steps, 1.0))
            lr_scheduler_config = {
                'scheduler': lr_scheduler,
                'interval': 'step',
            }

            return [optimizer], [lr_scheduler_config]

        return optimizer

def sample_test_images(diffusion_sampler, model, solver='second_order'):
    model.eval()
    with torch.no_grad():
        sampled_images = diffusion_sampler(model, n_classes=10, solver=solver)
    sampled_images = torch.clamp(sampled_images, 0, 1).cpu()
    grid = torchvision.utils.make_grid(sampled_images, nrow=5)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(f"test_samples_{solver}.png")

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='data/configs/config.yaml')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = Config(**config_dict['config'])
    T = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = CIFAR10('data/datasets/', download=True, transform=T)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=16)
    wandb_logger = WandbLogger(project='image-diffusion')
    
    if config.continue_from_checkpoint is not None:
        model = PLImageDiffusionModule.load_from_checkpoint(config.continue_from_checkpoint)
    else:
        model = PLImageDiffusionModule(config)

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath="image_diffusion/checkpoints", save_last=True)
    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(max_epochs=config.num_epochs, logger=wandb_logger, default_root_dir="image_diffusion/checkpoints", callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(model, train_dataloaders=train_loader)

def main():
    train()

    # print('Loading model...')
    # model = PLImageDiffusionModule.load_from_checkpoint("last_ckpt.ckpt", )
    # sampler = ImageDiffusionSampler(noise_steps=100)
    # print('Sampling images...')
    # for solver in ['first_order', 'second_order', 'stochastic_solver']:
    #     sample_test_images(sampler, model.model, solver=solver)
    # print('Done.')


if __name__ == "__main__":
    main()