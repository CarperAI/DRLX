from drlx.configs import SamplerConfig

class Sampler:
    def __init__(self, config : SamplerConfig):
        self.config = config

    def get_scalings(x, sigma):
        # TODO : This is relevant only to Karras sigmas
        sigma_data = self.config.sigma_data
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / ((sigma_data ** 2 + sigma ** 2) ** 0.5)
        c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5


    def sample(self, denoiser, sample, steps):
        pass