import torch
import torch.nn as nn


class Song2Face(nn.Module):
    """
    Song2Face: Synthesizing Singing Facial Animation from Audio
    https://dl.acm.org/doi/10.1145/3410700.3425435.
    """

    def __init__(self, n_verts: int, n_onehot: int):
        super().__init__()
        self.n_verts = n_verts
        self.n_onehot = n_onehot

        def conv_bn(in_channels, out_channels, kernel_size, stride, padding, bn=True):
            models = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            ]
            if bn:
                models.append(nn.BatchNorm2d(out_channels))
            models.append(nn.ReLU())
            return nn.Sequential(*models)

        def bi_lstm(input_size, hidden_size, num_layers=1):
            return nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                bidirectional=False,
                batch_first=True,
            )

        # 2 * Conv (5*1) + 3 * Conv (3*1) + 2 * BiLSTM
        self.vocal_encoder_nn = nn.Sequential(
            conv_bn(1, 72, (1, 5), (1, 2), (0, 2)),
            conv_bn(72, 108, (1, 5), (1, 2), (0, 2)),
            conv_bn(108, 162, (1, 3), (1, 2), (0, 1)),
            conv_bn(162, 243, (1, 3), (1, 2), (0, 1)),
            conv_bn(243, 256, (1, 3), (1, 2), (0, 1)),
        )
        self.vocal_encoder_lstm1 = bi_lstm(64, 256)
        self.vocal_encoder_lstm2 = bi_lstm(256, 256)

        self.output_net = nn.Sequential(
            nn.Linear(256 + n_onehot, 72),
            nn.Linear(72, 128),
            nn.Tanh(),
            nn.Linear(128, 50),
            nn.Linear(50, n_verts),
        )

        # 4 * Conv (1*3)

        self.regression_net = nn.Sequential(
            conv_bn(256, 256, (3, 1), (2, 1), (1, 0)),
            conv_bn(256, 256, (3, 1), (2, 1), (1, 0)),
            conv_bn(256, 256, (3, 1), (2, 1), (1, 0)),
            conv_bn(256, 256, (3, 1), (2, 1), (0, 0), False),
        )

    def forward(self, x, one_hot, template):
        bs = x.size(0)
        onehot_embedding = one_hot.repeat(1, 32).view(bs, 1, -1, 32)
        x = x.unsqueeze(1)
        x = torch.cat((x, onehot_embedding), 2)
        x = self.vocal_encoder_nn(x).squeeze(3)
        x, _ = self.vocal_encoder_lstm1(x)
        x, _ = self.vocal_encoder_lstm2(x)
        x = x.unsqueeze(3)
        x = torch.nn.functional.interpolate(x, size=(32, 1), mode="bilinear")
        x = self.regression_net(x)
        x = x.squeeze(3).squeeze(2)
        x = self.output_net(torch.cat((x, one_hot), 1))
        return x.view(bs, -1, 3) + template


if __name__ == "__main__":
    model = Song2Face(5023 * 3, 12).cuda()
    model.train()
    x = torch.randn(12, int(22000 * 0.52), requires_grad=True).cuda()
    one_hot = torch.randn(12, 12).cuda()
    template = torch.randn(12, 5023, 3).cuda()
    print(model(x, one_hot, template).size())
