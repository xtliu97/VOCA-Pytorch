import torch
import torch.nn as nn


class Voca(nn.Module):
    def __init__(self, n_verts: int, n_onehot: int):
        super().__init__()
        self.n_verts = n_verts
        self.n_onehot = n_onehot

        # self.feature_extractor = MFCCExtractor(
        #     sample_rate=22000,
        #     n_mfcc=16,
        #     out_dim=29,
        #     win_length=395 * 2,
        #     n_fft=2048,
        # )

        self.time_conv = nn.Sequential(
            nn.Conv2d(37, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64 + 8, 72),
            nn.Linear(72, 128),
            nn.Tanh(),
            nn.Linear(128, 50),
            nn.Linear(50, n_verts),
        )

    def forward(self, x, one_hot, template, **kwargs):
        bs = x.size(0)
        one_hot = one_hot[:, :8]
        onehot_embedding = one_hot.repeat(1, 16).view(bs, 1, -1, 16)
        x = x.unsqueeze(1)
        print(x.shape, onehot_embedding.shape)
        x = torch.cat((x, onehot_embedding), 2)  # [bs, 1, 37, 16]
        x = x.permute(0, 2, 3, 1)  # [bs, 37, 16,1]
        x = self.time_conv(x)
        x = torch.concat([x.view(bs, -1), one_hot], 1)  # [bs, 64 + 8]
        x = self.decoder(x)
        return x.view(bs, -1, 3) + template

    def predict(self, x, one_hot, template, **kwargs):
        return self(x, one_hot, template, **kwargs)
