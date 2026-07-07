import torch


class NoiseScheduler:

    def __init__(
        self,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02
    ):

        self.timesteps = timesteps


        self.beta = torch.linspace(
            beta_start,
            beta_end,
            timesteps
        )


        self.alpha = 1 - self.beta


        self.alpha_bar = torch.cumprod(
            self.alpha,
            dim=0
        )


    def add_noise(
        self,
        x0,
        noise,
        t
    ):

        sqrt_alpha_bar = torch.sqrt(
            self.alpha_bar[t]
        )


        sqrt_one_minus = torch.sqrt(
            1 - self.alpha_bar[t]
        )


        sqrt_alpha_bar = sqrt_alpha_bar.view(
            -1,1,1,1
        )


        sqrt_one_minus = sqrt_one_minus.view(
            -1,1,1,1
        )


        xt = (
            sqrt_alpha_bar * x0
            +
            sqrt_one_minus * noise
        )


        return xt