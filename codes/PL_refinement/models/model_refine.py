import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder_clip import VisionTransformer
from models.decoder_affordance import Affordance_Decoder
from models.SAM_decoder import SAM_Decoder_Simple


class ModelAGDsup(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, 
                 encoder_type, encoder_params,
                 img_size=224, patch_size=16, 
                 decoder_embed_dim=512, decoder_num_heads=16,
                 aff_decoder_depth=2, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, 
                 decoder_layer_scale_init_value=0.1,
                 init_std=0.02, pred_model_type="SAM",
                 pred_decoder_args={"mlp_dim":1024, "depth":2, "use_up":2, "use_additional_token":True},
                 use_text_feat=True,
                 margin=0.5
                 ):
        super().__init__()

        self.margin = margin
        self.encoder = VisionTransformer(
            input_resolution=img_size, patch_size=patch_size, **encoder_params)
        
        self.pred_decoder = SAM_Decoder_Simple(
            transformer_dim=decoder_embed_dim,
            activation=nn.GELU,
            **pred_decoder_args,
        )
          
        self.num_patches = self.encoder.num_patches
        self.patch_size = patch_size
        
    def forward(self, imgs, text_feat, ):
        _, x = self.encoder(imgs)
        
        v = text_feat.float().unsqueeze(1)
        aff_token = x[:, 0:1, ] + v
        
        pred_heatmap = self.pred_decoder(x, aff_token)
        
        return pred_heatmap
    