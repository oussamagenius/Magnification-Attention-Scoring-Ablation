"""
MagnificationAttentionMIL — Multi-Magnification Colposcopy Classification.

Architecture:
  1. Shared backbone (default: ResNet18) → 512-d feature vector per magnification
  2. Gated Attention Pooling (Ilse et al., ICML 2018) → attention weights α
  3. Weighted aggregation → patient-level feature
  4. Classification head → Normal/Abnormal probability

The gated attention weights α_m are directly interpretable as
magnification importance scores and are the central object of study
in the companion paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GatedAttentionPooling(nn.Module):
    """
    Gated Attention mechanism (Ilse et al., ICML 2018).

    Computes: a_m = W_a · (tanh(V · f_m) ⊙ σ(U · f_m))
              α_m = softmax(a_m)

    This is more expressive than simple attention because the sigmoid gate
    learns to suppress irrelevant features before the attention scoring.

    Args:
        feature_dim: Dimension of input features (from backbone)
        attention_dim: Hidden dimension of attention layers
        dropout: Dropout rate
    """

    def __init__(self, feature_dim: int, attention_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Sigmoid(),
        )
        self.attention_W = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> tuple:
        """
        Args:
            features: [B, num_magnifications, feature_dim]

        Returns:
            aggregated: [B, feature_dim] — weighted feature vector
            attention_weights: [B, num_magnifications] — α values
        """
        # Gated attention
        h_V = self.attention_V(features)        # [B, M, attention_dim]
        h_U = self.attention_U(features)        # [B, M, attention_dim]
        attention_scores = self.attention_W(self.dropout(h_V * h_U))  # [B, M, 1]
        attention_scores = attention_scores.squeeze(-1)               # [B, M]

        # Softmax over magnifications
        attention_weights = F.softmax(attention_scores, dim=1)        # [B, M]

        # Weighted aggregation
        # attention_weights: [B, M] -> [B, M, 1]
        aggregated = torch.sum(
            features * attention_weights.unsqueeze(-1), dim=1
        )  # [B, feature_dim]

        return aggregated, attention_weights


class SimpleAttentionPooling(nn.Module):
    """
    Simple attention mechanism (non-gated) for ablation comparison.

    Computes: a_m = W · tanh(V · f_m)
              α_m = softmax(a_m)
    """

    def __init__(self, feature_dim: int, attention_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> tuple:
        attention_scores = self.attention(features).squeeze(-1)  # [B, M]
        attention_weights = F.softmax(attention_scores, dim=1)   # [B, M]
        aggregated = torch.sum(
            features * attention_weights.unsqueeze(-1), dim=1
        )
        return aggregated, attention_weights


class MeanPooling(nn.Module):
    """
    Mean pooling for ablation (baseline without attention).
    Assigns equal uniform weight to all magnifications.
    """

    def forward(self, features: torch.Tensor) -> tuple:
        # features: [B, M, feature_dim]
        aggregated = torch.mean(features, dim=1)  # [B, feature_dim]
        # Equal weights: 1/M
        B, M, _ = features.shape
        attention_weights = torch.ones(B, M, device=features.device) / M
        return aggregated, attention_weights


class MagnificationAttentionMIL(nn.Module):
    """
    Full model: Backbone + Attention Pooling + Classifier.

    The backbone is shared across all magnifications (weight sharing),
    with differential learning rates:
    - Lower LR for pretrained backbone layers
    - Higher LR for attention + classifier head

    Optional backbone freezing via ``freeze_backbone_until=N`` freezes
    stem + blocks[0..N-1] to reduce trainable parameters and combat
    overfitting on small medical datasets.

    Args:
        cfg: Configuration dict containing model parameters
        num_magnifications: Number of magnification levels (default: 4)
    """

    def __init__(self, cfg: dict, num_magnifications: int = 4):
        super().__init__()
        self.cfg = cfg
        self.num_magnifications = num_magnifications

        backbone_name = cfg["training"]["backbone"]
        feature_dim = cfg["training"]["feature_dim"]
        attention_dim = cfg["model"]["attention_dim"]
        dropout = cfg["model"]["dropout"]
        dropout_attention = cfg["model"].get("dropout_attention", dropout)
        use_gated = cfg["model"]["gated_attention"]

        # 1. Shared backbone (supports timm models + medical foundation models)
        drop_path_rate = cfg["model"].get("drop_path_rate", 0.0)
        self.backbone = self._create_backbone(backbone_name, cfg, drop_path_rate)


        # Verify feature dimension (handles ViT returning unpooled tokens)
        with torch.no_grad():
            dummy = torch.randn(1, 3, cfg["data"]["image_size"], cfg["data"]["image_size"])
            out = self.backbone(dummy)
            if out.dim() == 3:
                self._pool_tokens = True   # ViT: pool tokens in forward
                actual_dim = out.shape[-1]
            else:
                self._pool_tokens = False
                actual_dim = out.shape[1]
            if actual_dim != feature_dim:
                print(f"WARNING: Expected feature_dim={feature_dim}, got {actual_dim}. Updating.")
                feature_dim = actual_dim

        # 2. Feature projection (optional dimensionality reduction)
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
        )

        # 3. Attention pooling
        pooling_type = cfg["model"].get("pooling", "attention")
        if pooling_type == "mean":
            self.attention = MeanPooling()
        elif use_gated:
            self.attention = GatedAttentionPooling(feature_dim, attention_dim, dropout_attention)
        else:
            self.attention = SimpleAttentionPooling(feature_dim, attention_dim, dropout_attention)

        # 4. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # Store attention weights for later retrieval
        self._last_attention_weights = None

        # V2: Freeze backbone layers if configured
        freeze_until = cfg["training"].get("freeze_backbone_until", -1)
        self._freeze_backbone(freeze_until)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, num_magnifications, 3, H, W]

        Returns:
            logits: [B] — raw logits (before sigmoid)
        """
        B, M, C, H, W = images.shape

        # Reshape to process all images through backbone at once
        images_flat = images.view(B * M, C, H, W)          # [B*M, 3, H, W]
        features_flat = self.backbone(images_flat)           # [B*M, feature_dim] or [B*M, tokens, dim] for ViT
        if getattr(self, "_pool_tokens", False):
            features_flat = features_flat.mean(dim=1)        # [B*M, feature_dim]
        features = features_flat.view(B, M, -1)             # [B, M, feature_dim]

        # Project features
        features = self.feature_proj(features)               # [B, M, feature_dim]

        # Attention pooling
        aggregated, attention_weights = self.attention(features)  # [B, feature_dim], [B, M]

        # Store for later retrieval
        if attention_weights is not None:
            self._last_attention_weights = attention_weights.detach()
        else:
            self._last_attention_weights = None

        # Classification
        logits = self.classifier(aggregated).squeeze(-1)     # [B]

        return logits

    def get_attention_weights(self) -> torch.Tensor:
        """
        Retrieve the attention weights from the last forward pass.

        Returns:
            Tensor of shape [B, num_magnifications] with values summing to 1.
        """
        if self._last_attention_weights is None:
            # Fallback to uniform if not captured
            return torch.ones(1, self.num_magnifications) / self.num_magnifications
        return self._last_attention_weights

    def _freeze_backbone(self, freeze_until: int):
        """
        Freezes the early layers of the backbone to prevent catastrophic forgetting.

        Args:
            freeze_until: block index to freeze up to. If >= 30, freezes entire backbone.
        """
        if freeze_until < 0:
            return  # No freezing

        frozen_count = 0

        if freeze_until >= 30:
            # Freeze entire backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
                frozen_count += param.numel()
        else:
            # EfficientNet specific
            if hasattr(self.backbone, 'conv_stem'):
                for param in self.backbone.conv_stem.parameters():
                    param.requires_grad = False
                    frozen_count += param.numel()
                for param in self.backbone.bn1.parameters():
                    param.requires_grad = False
                    frozen_count += param.numel()
            
            # ViT / DINOv2 specific
            if hasattr(self.backbone, 'patch_embed'):
                for param in self.backbone.patch_embed.parameters():
                    param.requires_grad = False
                    frozen_count += param.numel()

            # Freeze blocks[0..freeze_until-1] (timm uses .blocks for both)
            if hasattr(self.backbone, 'blocks'):
                for i, block in enumerate(self.backbone.blocks):
                    if i < freeze_until:
                        for param in block.parameters():
                            param.requires_grad = False
                            frozen_count += param.numel()
                            
            # ConvNeXt specific
            if hasattr(self.backbone, 'stem'):
                for param in self.backbone.stem.parameters():
                    param.requires_grad = False
                    frozen_count += param.numel()
            if hasattr(self.backbone, 'stages'):
                # ConvNeXt has 4 stages. freeze_until maps to stage index.
                for i, stage in enumerate(self.backbone.stages):
                    if i < freeze_until:
                        for param in stage.parameters():
                            param.requires_grad = False
                            frozen_count += param.numel()

            # Swin Transformer specific (e.g., CTransPath)
            if hasattr(self.backbone, 'layers') and not hasattr(self.backbone, 'stages'):
                for i, layer in enumerate(self.backbone.layers):
                    if i < freeze_until:
                        for param in layer.parameters():
                            param.requires_grad = False
                            frozen_count += param.numel()

        total_params = sum(p.numel() for p in self.backbone.parameters())
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(
            f"Backbone freezing: {frozen_count:,} params frozen "
            f"({frozen_count/total_params*100:.1f}%), "
            f"{trainable:,} trainable"
        )

    def get_parameter_groups(self, cfg: dict) -> list:
        """
        Create parameter groups with differential learning rates.

        Only includes backbone parameters that require gradients
        (unfrozen layers). Frozen layers are excluded from the optimizer
        entirely, saving memory and preventing accidental updates.

        Returns:
            List of dicts for the optimizer:
            - Group 0: Trainable backbone parameters (lower LR)
            - Group 1: Attention + Classifier parameters (higher LR)
        """
        # V2: only pass trainable backbone params
        backbone_params = [
            p for p in self.backbone.parameters() if p.requires_grad
        ]
        head_params = (
            list(self.feature_proj.parameters())
            + list(self.attention.parameters())
            + list(self.classifier.parameters())
        )

        return [
            {"params": backbone_params, "lr": float(cfg["training"]["lr_backbone"])},
            {"params": head_params, "lr": float(cfg["training"]["lr_head"])},
        ]

    # ── Backbone Factory Methods ─────────────────────────────────────────

    def _create_backbone(self, backbone_name: str, cfg: dict, drop_path_rate: float):
        """
        Factory method to create the feature extraction backbone.

        Supports:
        - Standard timm models (resnet18, efficientnet_b0, convnext_nano, etc.)
        - BiomedCLIP (ViT-B/16 pretrained on 15M medical image-text pairs)
        - CTransPath (Swin-Tiny pretrained on pathology images)
        - DINOv2 (ViT with dynamic image size)
        """
        name_lower = backbone_name.lower()

        if name_lower == "biomedclip":
            return self._load_biomedclip()

        elif name_lower == "ctranspath":
            return self._load_ctranspath(cfg, drop_path_rate)

        else:
            # Standard timm model
            kwargs = {}
            if "dinov2" in name_lower:
                kwargs["img_size"] = cfg["data"]["image_size"]
                kwargs["dynamic_img_size"] = True
            if drop_path_rate > 0.0:
                kwargs["drop_path_rate"] = drop_path_rate

            return timm.create_model(
                backbone_name,
                pretrained=True,
                num_classes=0,
                **kwargs
            )

    def _load_biomedclip(self):
        """
        Load BiomedCLIP's ViT-B/16 vision encoder (768-d features).

        BiomedCLIP (Microsoft, 2023) is pretrained on PMC-15M: 15 million
        biomedical figure-caption pairs from PubMed Central.

        Strategy: download the checkpoint via huggingface_hub and load ONLY
        the visual weights into a timm ViT-B/16.  This avoids building the
        text tower (PubMedBERT) which requires `transformers`.

        Requires: pip install huggingface_hub  (timm already a dependency)
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "BiomedCLIP requires huggingface_hub. Install with:\n"
                "  pip install huggingface_hub"
            )

        print("Downloading BiomedCLIP weights (vision-only) from HuggingFace hub...")
        ckpt_path = hf_hub_download(
            repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            filename="open_clip_pytorch_model.bin",
        )
        print(f"  Checkpoint at: {ckpt_path}")

        # Build vanilla timm ViT-B/16 (768-d) — same architecture as BiomedCLIP visual tower
        backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,       # we load BiomedCLIP weights below
            num_classes=0,          # remove head → 768-d CLS token
            dynamic_img_size=True,
        )

        # Load full CLIP checkpoint, extract only visual weights
        full_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(full_ckpt, dict) and "state_dict" in full_ckpt:
            full_ckpt = full_ckpt["state_dict"]

        # BiomedCLIP open_clip key format: "visual.trunk.*" (timm backend)
        # or "visual.*" (older open_clip).  Map to bare timm ViT-B/16 keys.
        visual_keys = {}
        for k, v in full_ckpt.items():
            if k.startswith("visual.trunk."):
                # open_clip >= 2.20 timm backend: keys are already timm-compatible
                visual_keys[k[len("visual.trunk."):]] = v
            elif k.startswith("visual."):
                new_key = k[len("visual."):]
                if new_key.startswith("proj") or new_key.startswith("head.proj"):
                    continue  # skip CLIP projection head
                visual_keys[new_key] = v

        if not visual_keys:
            raise RuntimeError(
                "Could not find 'visual.*' keys in BiomedCLIP checkpoint.\n"
                "First 10 keys found: " + str(list(full_ckpt.keys())[:10])
            )

        missing, unexpected = backbone.load_state_dict(visual_keys, strict=False)
        print(f"  Loaded {len(visual_keys)} visual weight tensors.")
        non_head_missing = [k for k in missing if "head" not in k]
        if non_head_missing:
            print(f"  WARNING — unexpected missing keys: {non_head_missing[:5]}")
        else:
            print(f"  (head.* missing keys are expected with num_classes=0 — OK)")

        del full_ckpt, visual_keys
        import gc
        gc.collect()
        print("BiomedCLIP ViT-B/16 loaded — no text tower, no transformers needed.")
        return backbone

    def _load_ctranspath(self, cfg: dict, drop_path_rate: float):
        """
        Load CTransPath backbone (Swin-Tiny, 768-d, pathology pretrained).

        CTransPath (Wang et al., 2022) is a Swin Transformer pretrained
        via unsupervised contrastive learning on ~15k whole slide images
        from TCGA + PAIP. Excels at tissue pattern recognition.

        If pretrained_weights path is provided in config, loads custom
        CTransPath weights. Otherwise falls back to ImageNet-pretrained
        Swin-Tiny (still useful but suboptimal for medical images).

        Download CTransPath weights from:
        https://github.com/Xiyue-Wang/TransPath
        """
        weights_path = cfg.get("training", {}).get("pretrained_weights", None)

        backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=(weights_path is None),  # ImageNet fallback
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )

        if weights_path:
            import os
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(
                    f"CTransPath weights not found: {weights_path}\n"
                    f"Download from: https://github.com/Xiyue-Wang/TransPath"
                )
            print(f"Loading CTransPath weights from: {weights_path}")
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            # Filter out classification head weights
            state_dict = {k: v for k, v in state_dict.items()
                          if not k.startswith("head.")}
            missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  CTransPath: {len(missing)} missing keys (expected for num_classes=0)")
            if unexpected:
                print(f"  CTransPath: {len(unexpected)} unexpected keys")
            print("CTransPath weights loaded successfully.")
        else:
            print(
                "WARNING: No CTransPath weights provided — using ImageNet-pretrained Swin-Tiny.\n"
                "  For best results, download CTransPath weights from:\n"
                "  https://github.com/Xiyue-Wang/TransPath\n"
                "  Then set training.pretrained_weights in config to the .pth file path."
            )

        return backbone
