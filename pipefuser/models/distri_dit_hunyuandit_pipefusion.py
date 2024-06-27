# from diffusers.models.attention_processor import Attention
from diffusers.models.attention import Attention
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.models.embeddings import PatchEmbed
from diffusers.models.transformers.hunyuan_transformer_2d import HunyuanDiT2DModel
from torch import distributed as dist, nn
import torch

from pipefuser.modules.base_module import BaseModule
from pipefuser.modules.dit.pipefusion import (
    DistriJointAttnPiP,
    DistriHunyuanDiT2DModel,
    DistriConv2dPiP,
    DistriPatchEmbed,
)

from .base_model import BaseModel
from ..utils import DistriConfig
from pipefuser.logger import init_logger

logger = init_logger(__name__)

from typing import Optional, Dict, Any


class HunyuanDiTPipeFusion(BaseModel):  # for Pipeline Parallelism
    def __init__(self, model: HunyuanDiT2DModel, distri_config: DistriConfig):
        assert isinstance(model, HunyuanDiT2DModel)
        model = DistriHunyuanDiT2DModel(model, distri_config)
        for name, module in model.named_modules():
            if isinstance(module, BaseModule):
                continue
            for subname, submodule in module.named_children():
                if isinstance(submodule, nn.Conv2d):
                    kernel_size = submodule.kernel_size
                    if kernel_size == (1, 1) or kernel_size == 1:
                        continue
                    wrapped_submodule = DistriConv2dPiP(
                        submodule, distri_config, is_first_layer=True
                    )
                    setattr(module, subname, wrapped_submodule)
                elif isinstance(submodule, PatchEmbed):
                    wrapped_submodule = DistriPatchEmbed(submodule, distri_config)
                    setattr(module, subname, wrapped_submodule)
                elif isinstance(submodule, Attention):
                    if subname == "attn1":  # self attention
                        wrapped_submodule = DistriJointAttnPiP(
                            submodule, distri_config
                        )
                        setattr(module, subname, wrapped_submodule)
        logger.info(
            f"Using pipeline parallelism, world_size: {distri_config.world_size} and n_device_per_batch: {distri_config.n_device_per_batch}"
        )
        super(HunyuanDiTPipeFusion, self).__init__(model, distri_config)

        self.batch_idx = 0

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states=None,
        text_embedding_mask=None,
        encoder_hidden_states_t5=None,
        text_embedding_mask_t5=None,
        image_meta_size=None,
        style=None,
        image_rotary_emb=None,
        return_dict=True,
    ):
        distri_config = self.distri_config

        # hidden_states.shape = [2, 4, 32, 32]
        # b, c, h, w = hidden_states.shape
        # b, c, h, w = sample.shape
        # assert (
        #     hidden_states is  None
        #     and encoder_hidden_states is None
        #     and text_embedding_mask is None
        #     # and encoder_attention_mask is None
        # )
        output = self.model(
                hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                text_embedding_mask=text_embedding_mask,
                encoder_hidden_states_t5=encoder_hidden_states_t5,
                text_embedding_mask_t5=text_embedding_mask_t5,
                image_meta_size=image_meta_size,
                style=style,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
                )[0]
        
        if return_dict:
            output = Transformer2DModelOutput(sample=output)
        else:
            output = (output,)

        self.counter += 1
        return output

    @property
    def add_embedding(self):
        return self.model.add_embedding