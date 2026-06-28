import torch
from torch import nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(output_channels)
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(output_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.shortcut(x), inplace=True)


class BottleNeck1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(16, channels // reduction)
        self.conv1 = nn.Conv1d(channels, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid)
        self.conv2 = nn.Conv1d(mid, mid, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(mid)
        self.conv3 = nn.Conv1d(mid, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))
        return F.relu(x + y, inplace=True)


class CNNModel(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(18, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResBlock1D(96, 96),
            ResBlock1D(96, 96),
            ResBlock1D(96, 96),
        )
        self.layer2 = nn.Sequential(
            ResBlock1D(96, 128, stride=2),
            ResBlock1D(128, 128),
            ResBlock1D(128, 128),
        )
        self.layer3 = nn.Sequential(
            ResBlock1D(128, 192, stride=2),
            ResBlock1D(192, 192),
            ResBlock1D(192, 192),
            BottleNeck1D(192),
        )
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 9, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
        )
        self.policy_head = nn.Linear(hidden, 235)
        self.type_head = nn.Linear(hidden, 8)
        self.value_head = nn.Linear(hidden, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _normalize_obs(self, input_dict):
        obs = input_dict['obs']['observation'].float().clone()
        if obs.shape[1] >= 18:
            obs[:, 6:7] = obs[:, 6:7] / 4.0
            obs[:, 7:8] = obs[:, 7:8] / 21.0
            obs[:, 9:17] = obs[:, 9:17] / 4.0
        return obs.reshape(obs.shape[0], obs.shape[1], 36)

    def forward_all(self, input_dict):
        self.train(mode=input_dict.get('is_training', False))
        x = self._normalize_obs(input_dict)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        hidden = self.neck(x)
        logits = self.policy_head(hidden)
        action_mask = input_dict['obs']['action_mask'].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return logits + inf_mask, self.type_head(hidden), self.value_head(hidden).squeeze(-1)

    def forward_policy_value(self, input_dict):
        logits, _, value = self.forward_all(input_dict)
        return logits, value

    def forward(self, input_dict):
        logits, _, _ = self.forward_all(input_dict)
        return logits
