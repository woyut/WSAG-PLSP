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
                 aff_decoder_depth=4,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, 
                 decoder_layer_scale_init_value=0.1,
                 init_std=0.02, pred_model_type="SAM",
                 pred_decoder_args={"mlp_dim":1024, "depth":2, "use_up":2, "use_additional_token":True},
                 margin=0.5
                 ):
        super().__init__()

        self.margin = margin
        self.encoder = VisionTransformer(
            input_resolution=img_size, patch_size=patch_size, **encoder_params)
        
        self.verb_fuser = Affordance_Decoder(
            num_patches=self.encoder.num_patches,
            decoder_embed_dim=decoder_embed_dim, regresser_depth=aff_decoder_depth, 
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None, 
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, init_values=decoder_layer_scale_init_value, init_std=init_std
        )
        
        self.pred_decoder = SAM_Decoder_Simple(
            transformer_dim=decoder_embed_dim,
            activation=nn.GELU,
            **pred_decoder_args,
        )
          
        self.num_patches = self.encoder.num_patches
        self.patch_size = patch_size
        
        self.exo_cls = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 36)
        )
        
        self.noun_transform = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )
        
        self.reason = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )
        
        
    def forward(self, imgs, text_feat, exo=None, exo_obj_mask=None, num_exo=1):
        _, x = self.encoder(imgs)
        
        v = text_feat.float().unsqueeze(1)
        pred_noun = self.noun_transform(x[:, 0:1, ].detach()) 
        pred_part = self.reason(torch.cat([pred_noun, v], dim=2))
        aff_token, _, _ = self.verb_fuser(x, pred_part+v)
        
        pred_heatmap = self.pred_decoder(x, aff_token)
        
        if exo is not None:
            # with torch.no_grad():
            _, exo = self.encoder(exo)
            exo_token = (exo[:, 1:] * exo_obj_mask).sum(dim=1)
            D = aff_token.shape[-1]
            aff_token_expand = aff_token.expand(-1, num_exo, -1).reshape(-1, D)
            sim_loss = torch.max(
                1 - F.cosine_similarity(aff_token_expand, exo_token.detach(), dim=1) - self.margin, 
                torch.zeros(len(exo_token)).to(x.device))
            return pred_heatmap, sim_loss, self.exo_cls(exo_token), pred_noun, pred_part
        else:
            return pred_heatmap, pred_noun, pred_part
    