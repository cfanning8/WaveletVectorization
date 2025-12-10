"""Neural network models for mammography pipeline."""

import re
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, roi_align
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.detection.image_list import ImageList
from torchvision.transforms import functional as F_tv

from . import config


def _is_swin(name: str) -> bool:
    """Check if backbone name is Swin Transformer."""
    return str(name).lower().startswith("swin")


def _swin_size_from_name(name: str) -> int:
    """Extract size from Swin Transformer name."""
    m = re.search(r"_([0-9]{2,4})(?:$|[^0-9])", str(name))
    return int(m.group(1)) if m else 224


class AttnMIL(nn.Module):
    """Attention-based Multiple Instance Learning head."""

    def __init__(self, in_dim: int, attn_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(attn_dim, 1, bias=False)
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, H: torch.Tensor):
        """Forward pass.
        
        Args:
            H: Instance features [B, P, D] where B=batch, P=patches, D=dim
            
        Returns:
            logit: Image-level logit [B]
            A: Attention weights [B, P]
            Z: Aggregated features [B, D]
        """
        A = self.U(torch.tanh(self.V(H))).squeeze(-1)
        A = torch.softmax(A, dim=1)
        Z = torch.sum(H * A.unsqueeze(-1), dim=1)
        logit = self.classifier(Z).squeeze(-1)
        return logit, A, Z


class Stage1MIL(nn.Module):
    """Stage-1 MIL model with backbone and attention."""

    def __init__(self, backbone_name: str = "convnext_tiny", pretrained=True, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.attn = AttnMIL(feat_dim)
        self._force_hw = (224, 224)
        self._use_gradient_checkpointing = use_gradient_checkpointing
        
        if use_gradient_checkpointing and hasattr(self.backbone, 'set_gradient_checkpointing'):
            self.backbone.set_gradient_checkpointing(True)
            print(f"[Stage1MIL] Gradient checkpointing enabled on {backbone_name} backbone")

    def forward_instances(self, x: torch.Tensor):
        """Forward pass for individual instances."""
        if x.shape[1] != 1:
            raise ValueError("Stage1 expects grayscale input [N,1,H,W].")
        
        if self._force_hw is not None and x.shape[-2:] != self._force_hw:
            x = F.interpolate(x, size=self._force_hw, mode="bilinear", align_corners=False)
        
        x3 = x.repeat(1, 3, 1, 1)
        
        return self.backbone(x3)

    def forward(self, bag: torch.Tensor):
        """Forward pass for bag of patches.
        
        Args:
            bag: Bag of patches [B, P, C, H, W]
            
        Returns:
            logit: Image-level logit [B]
            A: Attention weights [B, P]
            Z: Aggregated features [B, D]
        """
        B, P, C, H, W = bag.shape
        
        bags_flat = bag.view(B * P, C, H, W)
        E_flat = self.forward_instances(bags_flat)
        E = E_flat.view(B, P, -1)
        
        logit, A, Z = self.attn(E)
        return logit, A, Z

class TimmBackboneWithFPN(nn.Module):
    """Backbone wrapper with FPN and FiLM modulation.
    
    Accepts input with C∈{1 (gray), 9 (gray+8 topo)}.
    If C==9, uses the 8 topo channels to predict per-level (γ,β) and modulates features.
    """

    def __init__(self, backbone_name: str, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name

        # Stem to map arbitrary input C→3 for pretrained backbones
        self.conv1to3 = nn.Conv2d(1, 3, kernel_size=1)
        self.conv9to3 = nn.Conv2d(9, 3, kernel_size=1)  # for gray+8 topo

        # Backbone with feature extraction
        try:
            self.body = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(1, 2, 3, 4),
            )
        except Exception:
            self.body = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )

        if _is_swin(backbone_name):
            for m in self.body.modules():
                if hasattr(m, "strict_img_size"):
                    try:
                        m.strict_img_size = False
                    except Exception:
                        pass
                if hasattr(m, "dynamic_img_pad"):
                    try:
                        m.dynamic_img_pad = True
                    except Exception:
                        pass

        self.in_channels_list = list(self.body.feature_info.channels())
        assert len(self.in_channels_list) == 4, (
            f"Backbone '{backbone_name}' must provide 4 feature maps; "
            f"got {len(self.in_channels_list)} with channels {self.in_channels_list}."
        )
        self.fpn_names_in = [str(i) for i in range(len(self.in_channels_list))]

        self._out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            self.in_channels_list, self._out_channels, extra_blocks=None
        )

        # FiLM topo encoder
        self.topo_present_flag = False
        self.topo_stem = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.level_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(32, 2 * int(c))) for c in self.in_channels_list]
        )

    def _apply_film(self, feats_list: List[torch.Tensor], topo_8: torch.Tensor):
        """Apply FiLM modulation to feature maps."""
        B = topo_8.shape[0]
        topo_embed = self.topo_stem(topo_8)
        
        t_vec = F.adaptive_avg_pool2d(topo_embed, output_size=1).view(B, 32)
        
        out_feats = []
        for i, f in enumerate(feats_list):
            Bi, Ci, Hi, Wi = f.shape
            gb = self.level_heads[i](t_vec)
            gamma, beta = torch.split(gb, Ci, dim=1)
            gamma = torch.tanh(gamma)
            f = (
                f * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1))
                + beta.unsqueeze(-1).unsqueeze(-1)
            )
            out_feats.append(f)
        return out_feats

    def forward(self, x: torch.Tensor):
        """Forward pass.
        
        Args:
            x: Input tensor [B,C,H,W] where C∈{1,9}
            
        Returns:
            FPN output dictionary
        """
        topo_8 = None
        if x.shape[1] == 9:
            topo_8 = x[:, 1:9, :, :]
            x = self.conv9to3(x)
            self.topo_present_flag = True
        elif x.shape[1] == 1:
            x = self.conv1to3(x)
            self.topo_present_flag = False
        elif x.shape[1] == 3:
            self.topo_present_flag = False
        else:
            raise ValueError(f"Unexpected channel count: {x.shape[1]}")

        xs = self.body(x)

        feats_nchw = []
        for i, f in enumerate(xs):
            c_expect = int(self.in_channels_list[i])
            if f.dim() != 4:
                raise ValueError(
                    f"Feature map {i} is not 4D: got {tuple(f.shape)}"
                )
            if f.shape[1] == c_expect:
                f_nchw = f
            elif f.shape[-1] == c_expect:
                f_nchw = f.permute(0, 3, 1, 2).contiguous()
            else:
                raise ValueError(
                    f"Feature map {i} has shape {tuple(f.shape)}; "
                    f"expected channels={c_expect} in dim=1 or dim=-1."
                )
            feats_nchw.append(f_nchw)

        if self.topo_present_flag and topo_8 is not None:
            feats_nchw = self._apply_film(feats_nchw, topo_8)

        feats = {k: f for k, f in zip(self.fpn_names_in, feats_nchw)}
        out = self.fpn(feats)
        return out

    @property
    def out_channels(self):
        return self._out_channels

    @out_channels.setter
    def out_channels(self, v):
        self._out_channels = v


def build_faster_rcnn(backbone_name: str, pretrained_backbone=True) -> FasterRCNN:
    """Build Faster R-CNN model with custom backbone."""
    bb = TimmBackboneWithFPN(backbone_name, pretrained=pretrained_backbone)

    fpn_out_names = ["0", "1", "2", "3"]
    n_lvls = len(fpn_out_names)
    assert n_lvls >= 4, f"FPN returned {n_lvls} levels; expected >=4."

    anchor_sizes = tuple((32 * (2 ** i),) for i in range(n_lvls))
    aspect_ratios = tuple((0.5, 1.0, 2.0) for _ in range(n_lvls))

    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=fpn_out_names, output_size=7, sampling_ratio=2
    )

    bb.fpn_names = fpn_out_names

    model = FasterRCNN(
        bb,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )

    if _is_swin(backbone_name):
        s = _swin_size_from_name(backbone_name)
        model.transform.min_size = (s,)
        model.transform.max_size = s

    return model


def transplant_stage1_to_stage2(stage1: Stage1MIL, stage2):
    """Transfer weights from Stage-1 to Stage-2 backbone."""
    from .training import _normalize_state_dict_keys
    
    target_body = (
        stage2.det.backbone.body if hasattr(stage2, "det") else stage2.backbone.body
    )
    s1_raw = stage1.backbone.state_dict()
    s1 = _normalize_state_dict_keys(s1_raw)
    s2 = target_body.state_dict()
    copy_keys = [k for k in s1.keys() if k in s2]
    if not copy_keys:
        map1 = {k.split("module.")[-1]: k for k in s1.keys()}
        map2 = {k.split("module.")[-1]: k for k in s2.keys()}
        common = set(map1.keys()) & set(map2.keys())
        for ck in common:
            s2[map2[ck]] = s1[map1[ck]]
        target_body.load_state_dict(s2, strict=False)
    else:
        for k in copy_keys:
            s2[k] = s1[k]
        target_body.load_state_dict(s2, strict=False)



class FasterRCNNWithAttnMIL(nn.Module):
    """Faster R-CNN with attention-based MIL over RoI features."""

    def __init__(
        self,
        base: FasterRCNN,
        attn_dim: int = 128,
        mil_loss_weight: float = 1.0,
        topo_roi_dim: int = 64,
    ):
        super().__init__()
        self.det = base
        
        if hasattr(self.det.roi_heads.box_head, "fc7"):
            rep_dim = self.det.roi_heads.box_head.fc7.out_features
        else:
            rep_dim = self.det.roi_heads.box_predictor.cls_score.in_features

        self.topo_roi_dim = int(topo_roi_dim)
        self.topo_roi_stem = nn.Sequential(
            nn.Conv2d(8, self.topo_roi_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mil_head = AttnMIL(rep_dim + self.topo_roi_dim, attn_dim)
        self.mil_loss_weight = float(mil_loss_weight)

        self._swin_multiple = None
        if _is_swin(self.det.backbone.backbone_name):
            self._swin_multiple = self._infer_swin_multiple(self.det.backbone.body)

    @staticmethod
    def _infer_swin_multiple(body) -> int:
        """Infer Swin Transformer padding multiple."""
        ps = 4
        if hasattr(body, "patch_embed") and hasattr(body.patch_embed, "patch_size"):
            p = body.patch_embed.patch_size
            ps = int(p[0] if isinstance(p, (tuple, list)) else p)
        ws = 7
        for m in body.modules():
            if hasattr(m, "window_size"):
                w = m.window_size
                ws = int(w[0] if isinstance(w, (tuple, list)) else w)
                break
        return int(ps * ws)

    @staticmethod
    def _pad_to_multiple(image_list: ImageList, multiple: int) -> ImageList:
        """Pad image list to multiple of given value."""
        t = image_list.tensors
        B, C, H, W = t.shape
        Hpad = ((H + multiple - 1) // multiple) * multiple
        Wpad = ((W + multiple - 1) // multiple) * multiple
        if Hpad == H and Wpad == W:
            return image_list
        t2 = F.pad(t, (0, Wpad - W, 0, Hpad - H))
        return ImageList(t2, image_list.image_sizes)

    def _clear_swin_cached_masks(self):
        """Clear Swin Transformer cached attention masks."""
        for m in self.det.backbone.body.modules():
            if hasattr(m, "attn_mask"):
                setattr(m, "attn_mask", None)

    def _roi_feats_and_detections(self, images, targets=None):
        """Extract RoI features and run detection."""
        original_image_sizes = [img.shape[-2:] for img in images]
        
        has_9channel = any(img.shape[0] == 9 for img in images)
        if has_9channel:
            gray_images = [img[0:1] for img in images]
            topo_images = [img[1:9] for img in images]
            
            gray_list, targets = self.det.transform(gray_images, targets)
            
            gray_tensors = gray_list.tensors
            B, C_gray, H_transformed, W_transformed = gray_tensors.shape
            
            if C_gray == 3:
                gray_tensors = gray_tensors[:, 0:1, :, :]
            
            topo_tensors = []
            for i, topo in enumerate(topo_images):
                topo_resized = F_tv.resize(
                    topo.unsqueeze(0),
                    [H_transformed, W_transformed],
                    interpolation=F_tv.InterpolationMode.NEAREST
                ).squeeze(0)
                topo_tensors.append(topo_resized)
            
            topo_batch = torch.stack(topo_tensors, dim=0)
            
            combined = torch.cat([gray_tensors, topo_batch], dim=1)
            
            images_list = ImageList(combined, gray_list.image_sizes)
        else:
            images_list, targets = self.det.transform(images, targets)

        if _is_swin(self.det.backbone.backbone_name):
            mult = self._swin_multiple or 28
            images_list = self._pad_to_multiple(images_list, mult)
            self._clear_swin_cached_masks()

        topo_after = None
        if images_list.tensors.shape[1] >= 9:
            topo_after = images_list.tensors[:, 1:9, :, :]

        features = self.det.backbone(images_list.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}
        proposals, proposal_losses = self.det.rpn(images_list, features, targets)
        detections, detector_losses = self.det.roi_heads(
            features, proposals, images_list.image_sizes, targets
        )

        box_feats = self.det.roi_heads.box_roi_pool(
            features, proposals, images_list.image_sizes
        )
        box_feats = self.det.roi_heads.box_head(box_feats)

        if topo_after is not None:
            topo_rois = roi_align(
                topo_after,
                proposals,
                output_size=(7, 7),
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=True,
            )
            topo_feats_map = self.topo_roi_stem(topo_rois)
            topo_vec = F.adaptive_avg_pool2d(topo_feats_map, 1).view(
                topo_feats_map.size(0), -1
            )
        else:
            topo_vec = torch.zeros(
                (box_feats.size(0), self.topo_roi_dim),
                device=box_feats.device,
                dtype=box_feats.dtype,
            )

        mil_feats = torch.cat([box_feats, topo_vec], dim=1)

        num_per_img = [len(p) for p in proposals]
        split_feats = list(mil_feats.split(num_per_img, dim=0))
        detections = self.det.transform.postprocess(
            detections, images_list.image_sizes, original_image_sizes
        )
        return split_feats, detections, proposal_losses, detector_losses

    def forward(self, images, targets=None, image_labels=None):
        """Forward pass.
        
        Args:
            images: List of image tensors [C,H,W]
            targets: List of target dictionaries (for training)
            image_labels: Image-level labels (for training)
            
        Returns:
            If training: (losses, detections, image_probs)
            If inference: (detections, image_probs)
        """
        split_feats, detections, prop_losses, det_losses = self._roi_feats_and_detections(
            images, targets
        )

        device = images[0].device if images else config.DEVICE
        image_logits = []
        for H in split_feats:
            if H.numel() == 0:
                image_logits.append(torch.tensor(-10.0, device=device))
            else:
                logit, _, _ = self.mil_head(H.unsqueeze(0))
                image_logits.append(logit.squeeze(0))
        image_logits = (
            torch.stack(image_logits)
            if len(image_logits)
            else torch.empty(0, device=device)
        )
        image_probs = torch.sigmoid(image_logits)

        losses = {}
        losses.update(prop_losses)
        losses.update(det_losses)

        if self.training:
            assert (
                image_labels is not None
            ), "Stage-2 MIL requires image_labels during training."
            y = torch.as_tensor(
                image_labels, dtype=torch.float32, device=image_logits.device
            )
            loss_mil = F.binary_cross_entropy_with_logits(image_logits, y)
            losses["loss_mil"] = self.mil_loss_weight * loss_mil
            return losses, detections, image_probs
        else:
            return detections, image_probs

