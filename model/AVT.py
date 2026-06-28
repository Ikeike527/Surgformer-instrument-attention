# Copyright (c) Facebook, Inc. and its affiliates.
"""
Simplified AVT Model with single head for classification.
Removed future prediction and multiple branches for simplicity.
Backbone: (B, T, C) → AVTTransformer (GPT-2 based temporal modeling) → Classifier
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from timm.models.registry import register_model

# Try to import timm, but provide fallback if not available
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available, using PyTorch built-in models")


class TIMMModel(nn.Module):
    """TIMM model wrapper for video classification"""
    def __init__(self, model_type='vit_base_patch16_224_in21k', pretrained_model_path=None):
        super().__init__()
        if TIMM_AVAILABLE:
            model = timm.create_model(model_type, num_classes=0)
            self.model = model
        else:
            raise ImportError("timm is not available. Please install timm to use this model.")
        self.load_pretrained_backbone(pretrained_model_path)
    
    def load_pretrained_backbone(self, pretrained_model_path=None):
        """Load pretrained weights for the backbone"""
        if pretrained_model_path:
            print(f"Loading pretrained backbone from {pretrained_model_path}")
            try:
                checkpoint = torch.load(pretrained_model_path, map_location="cpu")
                
                # Handle different checkpoint formats
                if "model_state" in checkpoint:
                    checkpoint = checkpoint["model_state"]
                elif "model" in checkpoint:
                    checkpoint = checkpoint["model"]
                
                # Load only backbone weights
                backbone_state_dict = {}
                for k, v in checkpoint.items():
                    if k.startswith("backbone."):
                        backbone_state_dict[k[9:]] = v  # Remove "backbone." prefix
                    elif not k.startswith("head.") and not k.startswith("temporal_aggregator."):
                        # Assume other weights are backbone weights
                        backbone_state_dict[k] = v
                
                # Load to backbone model
                missing_keys, unexpected_keys = self.model.load_state_dict(backbone_state_dict, strict=False)
                print(f"Loaded pretrained backbone weights")
                if missing_keys:
                    print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")
                    
            except Exception as e:
                print(f"Warning: Could not load pretrained backbone weights: {e}")
                print("Continuing with random initialization")
        else:
            # Try to load from timm pretrained weights
            print("Loading ImageNet pretrained weights from timm")
            try:
                pretrained_model = timm.create_model('vit_base_patch16_224_in21k', num_classes=0, pretrained=True)
                self.model.load_state_dict(pretrained_model.state_dict())
                print("Successfully loaded ImageNet pretrained weights")
            except Exception as e:
                print(f"Warning: Could not load ImageNet pretrained weights: {e}")
                print("Continuing with random initialization")
    
    def forward(self, video, *args, **kwargs):
        """Process each frame separately"""
        # N, C, T, H, W
        batch_size = video.size(0) # N
        time_dim = video.size(2) # T
        video_flat = video.transpose(1, 2).flatten(0, 1) # N*T, C, H, W
        feats_flat = self.model(video_flat, *args, **kwargs) # N*T, C
        output = feats_flat.view((batch_size, time_dim) + feats_flat.shape[1:]).transpose(1, 2) # N, C, T
        return output
    
    @property
    def output_dim(self):
        """Return the output dimension of the backbone"""
        if hasattr(self.model, 'num_features'):
            return self.model.num_features
        elif hasattr(self.model, 'head'):
            return self.model.head.in_features
        else:
            # Default for ViT models
            return 768  # vit_base_patch16_224_in21k default feature dimension


class AVTTransformer(nn.Module):
    """AVT Transformer for temporal modeling (based on AVTh architecture)"""
    def __init__(self, in_features, num_layers=6, num_heads=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.inter_dim = dim_feedforward
        
        # Encoder: 768 → 2048 (like AVTh)
        self.encoder = nn.Linear(in_features, dim_feedforward, bias=False)
        
        # Decoder: 2048 → 768 (like AVTh)
        self.decoder = nn.Linear(dim_feedforward, in_features, bias=False)
        
        # GPT2 model (like AVTh)
        try:
            import transformers
            self.gpt_model = transformers.GPT2Model(
                transformers.GPT2Config(
                    n_embd=dim_feedforward,
                    vocab_size=1,  # using inputs_embeds, token vocab unused
                    use_cache=True,
                    n_positions=1000,  # explicit context length
                    n_layer=num_layers,
                    n_head=num_heads,
                    dropout=dropout
                )
            )
            # Remove the word token embedding since we're using our own encoder
            del self.gpt_model.wte
        except ImportError:
            print("Warning: transformers not available, falling back to standard Transformer")
            # Fallback to standard Transformer if transformers not available
            # Use a larger feedforward dimension (~4x) as is common practice
            ff_dim = 4 * dim_feedforward
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim_feedforward,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True
            )
            self.gpt_model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.use_gpt2 = False
        else:
            self.use_gpt2 = True
        
    def forward(self, feats):
        """
        Args:
            feats: (B, T, C) temporal features
        Returns:
            output: (B, C) aggregated features
            aux_losses: auxiliary losses dict
        """
        B, T, C = feats.shape
        
        # Encoder: 768 → 2048 (like AVTh)
        feats_encoded = self.encoder(feats)  # (B, T, 768) → (B, T, 2048)
        
        # Apply GPT2 (like AVTh)
        if self.use_gpt2:
            # Use GPT2 model
            # Follow AVTh: rely on GPT-2 internal positional embeddings via position_ids
            position_ids = torch.arange(0, T, dtype=torch.long, device=feats_encoded.device)
            position_ids = position_ids.unsqueeze(0).expand(B, T)
            outputs = self.gpt_model(inputs_embeds=feats_encoded, position_ids=position_ids)
            transformed = outputs.last_hidden_state  # (B, T, 2048)
        else:
            # Fallback to standard Transformer
            transformed = self.gpt_model(feats_encoded)  # (B, T, 2048)
        
        # Decode each time step first, then aggregate over time
        decoded = self.decoder(transformed)  # (B, T, 2048) → (B, T, 768)
        output = torch.mean(decoded, dim=1)  # (B, 768)
        
        return output, {}
    
    @property
    def output_dim(self):
        return self.in_features
    
    def no_weight_decay(self):
        """
        Return a list of parameter names that should not have weight decay applied.
        
        Returns:
            list: Parameter names to skip weight decay
        """
        # For AVTTransformer, typically bias terms don't have weight decay
        skip_list = []
        
        # Add bias parameters
        if hasattr(self.encoder, 'bias') and self.encoder.bias is not None:
            skip_list.append('encoder.bias')
        if hasattr(self.decoder, 'bias') and self.decoder.bias is not None:
            skip_list.append('decoder.bias')
        
        return skip_list


class MeanAggregator(nn.Module):
    """Mean temporal aggregation"""
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

    def forward(self, feats):
        """feats: B, T, C dimensional input"""
        return torch.mean(feats, dim=1), {}

    @property
    def output_dim(self):
        return self.in_features
    
    def no_weight_decay(self):
        """
        Return a list of parameter names that should not have weight decay applied.
        
        Returns:
            list: Parameter names to skip weight decay
        """
        # MeanAggregator has no parameters, so return empty list
        return []


class SimplifiedAVTModel(nn.Module):
    """
    Simplified AVT Model with single classification head.
    Removed future prediction and multiple branches.
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Dictionary containing model parameters
                Required keys:
                - backbone_type: 'timm' or other backbone types
                - backbone_model_type: model type for TIMM
                - temporal_aggregator_type: 'avt' or 'mean'
                - num_classes: number of classes for classification
                - intermediate_featdim: intermediate feature dimension
                - backbone_dim: backbone output dimension
                - backbone_last_n_modules_to_drop: number of backbone layers to drop
                - dropout: dropout rate
                - bn_eps: batch norm epsilon
                - bn_mom: batch norm momentum
        """
        super().__init__()
        
        # Store config for reference
        self.config = config
        
        # Initialize backbone
        self.backbone = self._create_backbone()
        
        # Initialize feature mapper
        self._init_feature_mapper()
        
        # Initialize temporal aggregator
        self._init_temporal_aggregator()
        
        # Initialize dropout
        self.dropout = nn.Dropout(config['dropout'])
        
        # Initialize classifier
        self._init_classifier()
        
        # Initialize compatibility attributes for training code
        self._init_compatibility_attrs()
        
        # Initialize weights and batch norm parameters
        self._initialize_weights()
        self._set_bn_params(config['bn_eps'], config['bn_mom'])
    
    def _create_backbone(self):
        """Create backbone network directly"""
        if self.config['backbone_type'] == 'timm':
            backbone = TIMMModel(
                model_type=self.config['backbone_model_type'],
                pretrained_model_path=self.config.get('backbone_pretrained_path'),
            )
        else:
            raise ValueError(f"Unsupported backbone type: {self.config['backbone_type']}")
        
        return backbone
    
    def _init_feature_mapper(self):
        """Initialize feature dimension mapper"""
        # Determine backbone output dimension
        if (hasattr(self.backbone, 'output_dim')):
            backbone_dim = self.backbone.output_dim
        else:
            backbone_dim = self.config['backbone_dim']
        
        # Set intermediate feature dimension
        if self.config['intermediate_featdim'] is None:
            self.config['intermediate_featdim'] = backbone_dim
        
        # Create mapper if dimensions don't match
        self.mapper_to_inter = None
        if backbone_dim != self.config['intermediate_featdim']:
            self.mapper_to_inter = nn.Linear(
                backbone_dim,
                self.config['intermediate_featdim'],
                bias=False
            )
    
    def _init_temporal_aggregator(self):
        """Initialize temporal aggregator"""
        if self.config['temporal_aggregator_type'] == 'avt':
            self.temporal_aggregator = AVTTransformer(
                in_features=self.config['intermediate_featdim'],
                num_layers=self.config.get('avt_num_layers', 6),
                num_heads=self.config.get('avt_num_heads', 8),
                dim_feedforward=self.config.get('avt_dim_feedforward', 2048),
                dropout=self.config.get('avt_dropout', 0.1)
            )
        elif self.config['temporal_aggregator_type'] == 'mean':
            self.temporal_aggregator = MeanAggregator(self.config['intermediate_featdim'])
        else:
            raise ValueError(f"Unsupported temporal aggregator type: {self.config['temporal_aggregator_type']}")
    
    def _init_classifier(self):
        """Initialize single classifier"""
        self.classifier = nn.Linear(self.temporal_aggregator.output_dim, self.config['num_classes'], bias=True)
    
    def _init_compatibility_attrs(self):
        """Initialize compatibility attributes for training code"""
        # Create patch_embed compatibility object
        class MockPatchEmbed:
            def __init__(self, patch_size):
                self.patch_size = patch_size
                # Calculate num_patches for 224x224 input with given patch size
                self.num_patches = (224 // patch_size[0]) * (224 // patch_size[1])
        
        # Create pos_embed compatibility object
        class MockPosEmbed:
            def __init__(self):
                self.shape = (1, 197, 768)  # Default ViT-B shape
        
        # For ViT models, patch size is typically (16, 16) for patch16 models
        if 'patch16' in self.config['backbone_model_type']:
            self.patch_embed = MockPatchEmbed((16, 16))
        elif 'patch32' in self.config['backbone_model_type']:
            self.patch_embed = MockPatchEmbed((32, 32))
        else:
            # Default to patch16
            self.patch_embed = MockPatchEmbed((16, 16))
        
        # Set pos_embed
        self.pos_embed = MockPosEmbed()
    
    def get_num_layers(self):
        """
        Get the total number of layers for layer decay learning rate.
        This is used by the training framework for layer-wise learning rate decay.
        
        Returns:
            int: Total number of layers (only backbone layers for layer decay)
        """
        # Only count backbone layers for layer decay
        # ViT typically has 12 layers for ViT-B
        backbone_layers = 12  # Default for ViT-B
        
        # Note: We only apply layer decay to the backbone (ViT) layers
        # Temporal aggregator and classifier use uniform learning rate
        return backbone_layers
    
    def no_weight_decay(self):
        """
        Return a list of parameter names that should not have weight decay applied.
        This is used by the optimizer.
        
        Returns:
            list: Parameter names to skip weight decay
        """
        # Common parameters that typically don't have weight decay
        skip_list = []
        
        # Add backbone-specific parameters if available
        if hasattr(self.backbone, 'no_weight_decay'):
            backbone_skip = self.backbone.no_weight_decay()
            # Prefix with 'backbone.' to match the actual parameter names
            skip_list.extend([f'backbone.{name}' for name in backbone_skip])
        
        # Add temporal aggregator parameters if available
        if hasattr(self.temporal_aggregator, 'no_weight_decay'):
            temporal_skip = self.temporal_aggregator.no_weight_decay()
            # Prefix with 'temporal_aggregator.' to match the actual parameter names
            skip_list.extend([f'temporal_aggregator.{name}' for name in temporal_skip])
        
        # Add common parameters that typically don't have weight decay
        skip_list.extend([
            'classifier.bias',  # Classifier bias
            'mapper_to_inter.bias' if self.mapper_to_inter else None,  # Mapper bias if exists
        ])
        
        # Remove None values
        skip_list = [name for name in skip_list if name is not None]
        
        return skip_list
    
    def _initialize_weights(self):
        """Initialize model weights for Transformer-based architecture"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier initialization for linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                # Initialize LayerNorm parameters
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Dropout):
                # Dropout doesn't need initialization
                pass
    
    def _set_bn_params(self, bn_eps=1e-3, bn_mom=0.1):
        """Set normalization parameters for Transformer architecture"""
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                module.eps = bn_eps
            # Note: LayerNorm doesn't use momentum parameter
    
    def forward(self, video: torch.Tensor):
        """
        Forward pass for video classification
        
        Args:
            video: Input video tensor (B, C, T, H, W) or (B, 1, C, T, H, W)
            
        Returns:
            logits: Classification logits (B, num_classes)
        """
        # Handle different input shapes
        if video.ndim == 6 and video.size(1) == 1:
            video = video.squeeze(1)  # Remove extra dimension
        
        if video.ndim != 5:
            raise ValueError(f'Expected video shape (B, C, T, H, W), got {video.shape}')
        
        # Backbone feature extraction
        feats = self.backbone(video)  # (N, C, T)
        
        # Rearrange dimensions: B,C,T -> B,T,C
        feats = feats.permute((0, 2, 1))  # (B, T, C)
        
        # Map to intermediate dimension if needed
        if feats.shape[-1] != self.config['intermediate_featdim']:
            assert self.mapper_to_inter is not None, (
                f'Backbone feature dimension {feats.shape} does not match '
                f'intermediate dimension {self.config["intermediate_featdim"]}. '
                f'Please set backbone_dim correctly.'
            )
            feats = self.mapper_to_inter(feats)
        
        # Temporal aggregation
        feats_agg, _ = self.temporal_aggregator(feats)  # (B, C)
        
        # Apply dropout and classifier
        feats_agg_drop = self.dropout(feats_agg)
        logits = self.classifier(feats_agg_drop)  # (B, num_classes)
        
        return logits


# Example configuration for simplified model
def create_simplified_config(num_classes: int = 7):
    """Create a simplified configuration for AVTModel"""
    config = {
        # Backbone configuration
        'backbone_type': 'timm',
        'backbone_model_type': 'vit_base_patch16_224_in21k',
        'backbone_pretrained_path': None,
        
        # Temporal aggregator configuration
        'temporal_aggregator_type': 'avt',  # Use AVT Transformer (like AVTh)
        
        # AVT Transformer parameters (matching original AVTh)
        'avt_num_layers': 6,           # n_layer=6 from original config
        'avt_num_heads': 4,            # n_head=4 from original config  
        'avt_dim_feedforward': 2048,   # inter_dim=2048 from original config
        'avt_dropout': 0.1,
        
        # Model parameters
        'intermediate_featdim': 768,  # ViT-B output dimension
        'backbone_dim': 768,          # ViT-B output dimension
        'dropout': 0.2,               # Match original AVT dropout
        
        # Batch normalization parameters
        'bn_eps': 1e-3,
        'bn_mom': 0.1,
        
        # Classification
        'num_classes': num_classes,
    }
    
    return config

@register_model
def AVT(pretrained=False, pretrain_path=None, **kwargs):
    config = create_simplified_config(num_classes=kwargs['num_classes'])
    config['backbone_pretrained_path'] = pretrain_path if pretrained else None
    
    # Create simplified model
    model = SimplifiedAVTModel(config)
    return model

# Example usage
if __name__ == "__main__":
    # Create simplified configuration
    kwargs = {"nb_classes": 7}
    model = AVT(pretrained=True, pretrain_path=None, **kwargs)
    
    # Example input
    batch_size = 2
    channels = 3
    time_steps = 16
    height = 224
    width = 224
    
    video = torch.randn(batch_size, channels, time_steps, height, width)
    
    # Forward pass
    logits = model(video)
    
    print(f"Simplified AVT Model created successfully!")
    print(f"Input shape: {video.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
