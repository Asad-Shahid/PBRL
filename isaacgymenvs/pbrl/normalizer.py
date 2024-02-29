import torch
# https://github.com/supersglzc/pql/blob/master/pql/utils/torch_util.py


class Normalizer():
    def __init__(self, dim, epsilon=1e-4, device='cuda'):
        self.device = device
        self.mean = torch.zeros(dim, device=self.device)
        self.var = torch.ones(dim, device=self.device)
        self.epsilon = epsilon
        self.count = epsilon

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, x):
        out = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        return out

    def unnormalize(self, x):
        out = x * torch.sqrt(self.var + self.epsilon) + self.mean
        return out

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def state_dict(self):
        
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count
        }
    
    def load_state_dict(self, ckpt):
        self.mean = ckpt["mean"]
        self.var = ckpt["var"]
        self.count = ckpt["count"]
