import torch
import torch.nn as nn


class Audio2Mesh(nn.Module):
    """https://research.nvidia.com/sites/default/files/publications/karras2017siggraph-paper_0.pdf"""

    def __init__(self, n_verts: int, n_onehot: int):
        super().__init__()
        self.n_verts = n_verts
        self.n_onehot = n_onehot

        self.analysis_net = nn.Sequential(
            nn.Conv2d(1, 72, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Conv2d(72, 108, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(108),
            nn.ReLU(),
            nn.Conv2d(108, 162, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(162),
            nn.ReLU(),
            nn.Conv2d(162, 243, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(243),
            nn.ReLU(),
            nn.Conv2d(243, 256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.articulation_net = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(4, 1), stride=(4, 1)),
            nn.ReLU(),
        )

        self.output_net = nn.Sequential(
            nn.Linear(256 + n_onehot, 72),
            nn.Linear(72, 128),
            nn.Tanh(),
            nn.Linear(128, 50),
            nn.Linear(50, n_verts),
        )

    def forward(self, x, one_hot, template, **kwargs):
        bs = x.size(0)
        onehot_embedding = one_hot.repeat(1, 32).view(bs, 1, -1, 32)
        x = x.unsqueeze(1)
        # x = self.instance_norm(x)
        x = self.analysis_net(torch.cat((x, onehot_embedding), 2))
        x = self.articulation_net(x)
        x = x.view(x.size(0), -1)
        x = self.output_net(torch.cat((x, one_hot), 1))
        return x.view(bs, -1, 3) + template

    def predict(self, x, one_hot, template, **kwargs):
        return self(x, one_hot, template, **kwargs)
