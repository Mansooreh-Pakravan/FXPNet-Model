# modelClasses_21.py

import torch
import torch.nn as nn

class ConvNet21(nn.Module):
    def __init__(self,n_roi, n_classes=2):
        super(ConvNet21, self).__init__()
        # CNN Block 1 (کاملاً مثل ConvNet، فقط 21 به‌جای 246)
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=n_roi, out_channels=256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=2)
        )

        # CNN Block 2 (مثل نسخه اصلی، 256→512 با kernel=5)
        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=5, stride=1),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=5, stride=1),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=2)
        )

        # Dropout
        self.drop_out = nn.Dropout(p=0.5)
        # Output FC Layer
        self.fc1 = nn.Linear(512, 2)

    def forward(self, x):
        # x شکلش باید (B, 21, T) باشد
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # دقیقاً مثل کد اصلی: layer5 تعریف شده، ولی استفاده نمی‌شود
        # out = self.layer5(out)
        out = self.layer6(out)

        # Temporal Averaging روی محور زمان
        out = out.mean(axis=2)   # (B, 512)

        out = self.drop_out(out)
        out = self.fc1(out)      # (B, 2)

        return out




# fuzzy_prototype_net21_v3_light.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

# ----------------------------
# 1) Backbone سبک (ConvNet-like)
# ----------------------------
class ConvBackbone21Light(nn.Module):
    """
    ورودی: (B, n_roi, T)
    خروجی: h (B, feat_dim)
    """
    def __init__(self, n_roi: int, feat_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        # سبک‌تر از ConvNet21: 256->128->256
        self.conv1 = nn.Conv1d(n_roi, 128, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=1)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=5, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=2)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(256, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool1(F.relu(self.bn1(self.conv1(x))))
        h = self.pool2(F.relu(self.conv2(h)))
        h = self.pool3(F.relu(self.conv4(F.relu(self.conv3(h)))))

        # global mean over time
        h = h.mean(dim=2)              # (B, 256)
        h = self.dropout(h)
        h = self.out_proj(h)           # (B, feat_dim)
        return h


# ----------------------------
# 2) Fuzzy Prototype Layer (RBF membership)
# ----------------------------
class FuzzyPrototypeLayer(nn.Module):
    """
    prototypes: (K, d)
    membership: mu (B, K) in (0,1]
    """
    def __init__(self, n_prototypes: int, latent_dim: int, init_sigma: float = 1.0):
        super().__init__()
        self.K = n_prototypes
        self.d = latent_dim

        self.prototypes = nn.Parameter(torch.randn(n_prototypes, latent_dim) * 0.1)

        # sigma per prototype (learnable, positive)
        self.log_sigma = nn.Parameter(torch.log(torch.ones(n_prototypes) * init_sigma))

    def forward(self, z: torch.Tensor, return_details: bool = False):
        """
        z: (B, d)
        returns:
          mu: (B, K)
          dist2: (B, K)
        """
        # squared Euclidean distance
        # dist2[b,k] = ||z_b - p_k||^2
        z2 = (z ** 2).sum(dim=1, keepdim=True)           # (B,1)
        p2 = (self.prototypes ** 2).sum(dim=1).view(1,-1) # (1,K)
        zp = z @ self.prototypes.t()                     # (B,K)
        dist2 = z2 + p2 - 2 * zp                         # (B,K)

        sigma = torch.exp(self.log_sigma).view(1, -1) + 1e-6
        mu = torch.exp(-dist2 / (2.0 * sigma**2))        # (B,K)

        if not return_details:
            return mu

        details = {
            "dist2": dist2,
            "sigma": sigma.detach().clone(),
        }
        return mu, details


# ----------------------------
# 3) مدل نهایی: Backbone + Bottleneck + Proto + Residual Head
# ----------------------------
class FuzzyProtoNet21V3Light(nn.Module):
    """
    - backbone -> h (B, feat_dim)
    - bottleneck -> z (B, d)
    - fuzzy prototypes -> mu (B, K)
    - proto logits: sum mu of prototypes assigned to each class
    - residual linear head on z for accuracy safety
    """
    def __init__(
        self,
        n_roi: int,
        n_classes: int = 2,
        feat_dim: int = 256,
        latent_dim: int = 32,
        protos_per_class: int = 8,    # مثلا 8 پروتو برای هر کلاس => K=16
        dropout: float = 0.3,
        lambda_proto: float = 0.7,    # وزن proto logits در ترکیب
        init_sigma: float = 1.0,
        temperature: float = 1.0,     # برای کنترل sharpness در proto logits
    ):
        super().__init__()
        assert n_classes == 2, "برای سادگی این نسخه 2 کلاس (M/F) را فرض کرده است."

        self.n_roi = n_roi
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.protos_per_class = protos_per_class
        self.K = protos_per_class * n_classes
        self.lambda_proto = lambda_proto
        self.temperature = temperature

        self.backbone = ConvBackbone21Light(n_roi=n_roi, feat_dim=feat_dim, dropout=dropout)

        self.bottleneck = nn.Sequential(
            nn.Linear(feat_dim, latent_dim),
            nn.ReLU(),
        )

        self.proto = FuzzyPrototypeLayer(self.K, latent_dim, init_sigma=init_sigma)

        # prototype-class assignment: first block -> class0, second block -> class1
        # class 0 = Male, class 1 = Female (همان نگاشت شما)
        proto_class = torch.zeros(self.K, dtype=torch.long)
        proto_class[protos_per_class:] = 1
        self.register_buffer("proto_class", proto_class)

        # learnable weights for each prototype contribution to its class (optional, helps accuracy)
        self.proto_logit_w = nn.Parameter(torch.zeros(self.K))  # will be softplus -> positive

        # residual classifier (safety)
        self.linear_head = nn.Linear(latent_dim, n_classes)

    def forward(self, x: torch.Tensor, return_details: bool = False):
        """
        x: (B, n_roi, T)
        returns logits (B,2)
        """
        h = self.backbone(x)           # (B, feat_dim)
        z = self.bottleneck(h)         # (B, latent_dim)

        if return_details:
            mu, proto_details = self.proto(z, return_details=True)  # (B,K)
        else:
            mu = self.proto(z, return_details=False)
            proto_details = None

        # proto logits:
        # for each class, sum (w_k * mu_k) over prototypes of that class
        w = F.softplus(self.proto_logit_w).view(1, -1)            # (1,K), positive
        mu_w = mu * w                                             # (B,K)

        # masks per class
        mask0 = (self.proto_class == 0).view(1, -1).float()
        mask1 = (self.proto_class == 1).view(1, -1).float()

        logit0 = (mu_w * mask0).sum(dim=1, keepdim=True) / self.temperature
        logit1 = (mu_w * mask1).sum(dim=1, keepdim=True) / self.temperature
        logits_proto = torch.cat([logit0, logit1], dim=1)         # (B,2)

        logits_lin = self.linear_head(z)                          # (B,2)

        logits = self.lambda_proto * logits_proto + (1.0 - self.lambda_proto) * logits_lin

        if not return_details:
            return logits

        details: Dict[str, Any] = {
            "z": z,                          # (B,d)  (بدون detach برای reg)
            "mu": mu,                        # (B,K)
            "proto_class": self.proto_class, # (K,)
            "prototypes": self.proto.prototypes,  # (K,d)
            "proto_extra": proto_details,
            "logits_proto": logits_proto,
            "logits_lin": logits_lin,
        }
        return logits, details
