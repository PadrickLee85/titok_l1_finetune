"""This file contains the model definition of TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""

import torch
import torch.nn as nn
from einops import rearrange
import math

# from modeling.modules.base_model import BaseModel
from modeling.modules.blocks import TiTokEncoder, TiTokDecoder
from modeling.quantizer.quantizer import VectorQuantizer, DiagonalGaussianDistribution
from modeling.modules import BaseModel
from modeling.modules import ConvEncoder as Pixel_Encoder
from modeling.modules import ConvDecoder as Pixel_Decoder
from modeling.quantizer import LookupFreeQuantizer
import json
from omegaconf import OmegaConf
from pathlib import Path

from huggingface_hub import PyTorchModelHubMixin


def choose_vector_quantizer_class(config):
    if config.quantizer_type == "lookup":
        return SimpleVectorizer(
            config.codebook_size,
            config.token_size,
            config.commitment_cost,
            config.entropy_loss_weight,
            config.entropy_loss_temperature,
            config.entropy_gamma,
            config.get("use_l2_normalisation", False),
        )
    elif config.quantizer_type == "lookup-free":
        Pixel_Quantizer = LookupFreeQuantizer
        return Pixel_Quantizer(
            config.token_size,
            config.commitment_cost,
            config.entropy_loss_weight,
            config.entropy_loss_temperature,
            config.entropy_gamma,
        )
    elif config.quantizer_type == "vae":
        return NotImplementedError("Currently not supported. We welcome a well tested PR.")
    else:
        raise ValueError("Unknown vector quantizer class")


class ConvVQModel(BaseModel):
    def __init__(
        self,
        config,
    ):
        """ Initializes the convolutional VQ-VAE model.

        Args:
        """
        super().__init__()
        self.config = config
        self.encoder = Pixel_Encoder(self.config)
        self.decoder = Pixel_Decoder(self.config)
        self.quantize = choose_vector_quantizer_class(self.config)

        # Load pretrained weights
        self.load_state_dict(torch.load(config.pretrained_weight, map_location=torch.device("cpu")), strict=False)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        """ Encodes the input tensor, just with the encoder, not conclude quantize.

        Args:
            x -> torch.Tensor: The input tensor.

        Returns:
            z : the float feature before quantize, which is also what the titok's [encoder/quantize/decoder]'s output wanna align.
        """
        z = self.encoder(x)  #  b c h w

        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor):
        """ Decodes latent representation, i.e. runs the quantizer and decoder.

        Args:
            z_quantized -> torch.Tensor: The quantized latent representation.
            decoded : the decoded image.
        """
        z_quantized, result_dict = self.quantize(z)
        decoded = self.decoder(z_quantized)
        return decoded

    @torch.no_grad()
    def decode_tokens(self, tokens: torch.Tensor):
        """ Decodes from tokens, i.e. runs the decoder after converting tokens to latent representations.

        Args:
            tokens -> torch.Tensor: The tokens.

        Returns:
            decoded/rec_images -> torch.Tensor: The decoded image.
        """
        # z_quantized = self.quantize.get_codebook_entry(tokens) # b l d c
        # import pdb;pdb.set_trace()
        z_quantized = tokens
        ##newadd
        depth_codebook = 4
        for depth in range(depth_codebook):
            z_quantized[:, :, depth, :] = z_quantized[:, :, depth, :]*(1/(2**depth))
        z_quantized = z_quantized.sum(dim=-2) # B L D C -> B L C
        
        ss = int(math.sqrt(float(z_quantized.size(1))))
        z_quantized = z_quantized.reshape(z_quantized.size(0), ss, ss, -1)
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        decoded = self.decode(z_quantized)

        rec_images = torch.clamp(decoded, 0.0, 1.0)
        return rec_images.detach()


class PretrainedTokenizer(nn.Module):
    def __init__(self, pretrained_weight):
        super().__init__()
        conf = OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "dropout": 0.0,
            "hidden_channels": 128,
            "num_channels": 3,
            "num_res_blocks": 2,
            "resolution": 256,
            "z_channels": 256})
        self.encoder = Pixel_Eecoder(conf)
        self.decoder = Pixel_Decoder(conf)
        self.quantize = Pixel_Quantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        # Load pretrained weights
        self.load_state_dict(torch.load(pretrained_weight, map_location=torch.device("cpu")), strict=True)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode(self, x):
        hidden_states = self.encoder(x)
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states)
        return codebook_indices.detach()
    
    @torch.no_grad()
    def decode(self, codes):
        quantized_states = self.quantize.get_codebook_entry(codes)
        rec_images = self.decoder(quantized_states)
        rec_images = torch.clamp(rec_images, 0.0, 1.0)
        return rec_images.detach()
    
    @torch.no_grad()
    def decode_tokens(self, codes):
        return self.decode(codes)


class TiTok(BaseModel, PyTorchModelHubMixin, tags=["arxiv:2406.07550", "image-tokenization"], repo_url="https://github.com/bytedance/1d-tokenizer", license="apache-2.0"):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        # This should be False for stage1 and True for stage2.
        self.finetune_decoder = config.model.vq_model.get("finetune_decoder", True)

        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        if self.quantize_mode not in ["vq", "lfq", "vae"]:
            raise ValueError(f"Unsupported quantize mode {self.quantize_mode}.")
        
        if self.finetune_decoder and self.quantize_mode not in ["vq"]:
            raise ValueError("Only supprot finetune_decoder with vq quantization for now.")

        self.encoder = TiTokEncoder(config)
        self.decoder = TiTokDecoder(config)   # just make sure the output ffn's dim==10, which is same like maskbit tokenizer.encoder output dim
        
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)

        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=config.model.vq_model.codebook_size,
                token_size=config.model.vq_model.token_size,
                commitment_cost=config.model.vq_model.commitment_cost,
                use_l2_norm=config.model.vq_model.use_l2_norm,)
        elif self.quantize_mode == "lfq":
            self.quantize = LookupFreeQuantizer(
                token_bits = config.model.vq_model.token_size,
                commitment_cost = config.model.vq_model.commitment_cost,
                entropy_loss_weight = config.model.vq_model.entropy_loss_weight,
                entropy_loss_temperature = config.model.vq_model.entropy_loss_temperature,
                entropy_gamma = config.model.vq_model.entropy_gamma,)
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError
        
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        if self.quantize_mode == "vq":
            z_quantized, result_dict = self.quantize(z)
        elif self.quantize_mode == "lfq":
            z_quantized, result_dict = self.quantize(z)
        elif self.quantize_mode == "vae":
            posteriors = self.quantize(z)
            z_quantized = posteriors.sample()
            result_dict = posteriors

        return z_quantized, result_dict
    
    def decode(self, z_quantized):
        decoded = self.decoder(z_quantized) # bsz 10 16 16
        return decoded
    
    def decode_tokens(self, tokens):
        if self.quantize_mode == "vq":
            tokens = tokens.squeeze(1)
            batch, seq_len = tokens.shape # B x N
            z_quantized = self.quantize.get_codebook_entry(
                tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
            z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        elif self.quantize_mode == "lfq":
            """ Decodes from tokens, i.e. runs the decoder after converting tokens to latent representations.

            Args:
                tokens -> torch.Tensor: The tokens.

            Returns:
                decoded -> torch.Tensor: The decoded image.
            """
            z_quantized = self.quantize.get_codebook_entry(tokens) # b l d c
            ##newadd
            depth_codebook = 4
            for depth in range(depth_codebook):
                z_quantized[:, :, depth, :] = z_quantized[:, :, depth, :]*(1/(2**depth))
            z_quantized = z_quantized.sum(dim=-2) # B L D C -> B L C

            ss = int(math.sqrt(float(z_quantized.size(1))))
            z_quantized = z_quantized.reshape(z_quantized.size(0), ss, ss, -1)
            z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        elif self.quantize_mode == "vae":
            z_quantized = tokens
        decoded = self.decode(z_quantized)
        return decoded
    
    def forward(self, x):
        z_quantized, result_dict = self.encode(x)
        decoded = self.decode(z_quantized)
        return decoded, result_dict