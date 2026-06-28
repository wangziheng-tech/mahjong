import torch
from torch import nn


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_se=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se = SqueezeExcite(channels) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.se(self.conv(x)))


class CNNModel(nn.Module):
    def __init__(self, channels=160, blocks=8, hidden=1024, dropout=0.20):
        super().__init__()
        layers = [
            nn.Conv2d(18, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks):
            layers.append(ResidualBlock(channels, use_se=True))
        self.features = nn.Sequential(*layers)
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 4 * 9, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.policy_head = nn.Linear(hidden, 235)
        self.type_head = nn.Linear(hidden, 8)
        self.value_head = nn.Linear(hidden, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _normalize_obs(self, input_dict):
        obs = input_dict["obs"]["observation"].float().clone()
        if obs.shape[1] >= 18:
            obs[:, 6:7] = obs[:, 6:7] / 4.0
            obs[:, 7:8] = obs[:, 7:8] / 21.0
            obs[:, 9:17] = obs[:, 9:17] / 4.0
        return obs

    def forward_all(self, input_dict):
        self.train(mode=input_dict.get("is_training", False))
        obs = self._normalize_obs(input_dict)
        hidden = self.neck(self.features(obs))
        logits = self.policy_head(hidden)
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return logits + inf_mask, self.type_head(hidden), self.value_head(hidden).squeeze(-1)

    def forward_policy_value(self, input_dict):
        logits, _, value = self.forward_all(input_dict)
        return logits, value

    def forward(self, input_dict):
        logits, _, _ = self.forward_all(input_dict)
        return logits
