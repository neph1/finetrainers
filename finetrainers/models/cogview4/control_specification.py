import os
from typing import Dict, Optional, Tuple

import safetensors.torch
import torch
from diffusers import AutoencoderKL, CogView4Pipeline, CogView4Transformer2DModel, FlowMatchEulerDiscreteScheduler

from ... import functional as FF
from ...typing import SchedulerType
from ..modeling_utils import ControlModelSpecification
from ..utils import DiagonalGaussianDistribution, _expand_linear_with_zeroed_weights
from .base_specification import CogView4ModelSpecification


class CogView4ControlModelSpecification(ControlModelSpecification, CogView4ModelSpecification):
    def load_diffusion_models(self, new_in_features: int) -> Dict[str, torch.nn.Module]:
        if self.transformer_id is not None:
            transformer = CogView4Transformer2DModel.from_pretrained(
                self.transformer_id,
                torch_dtype=self.transformer_dtype,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        else:
            transformer = CogView4Transformer2DModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=self.transformer_dtype,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )

        _expand_linear_with_zeroed_weights(transformer.patch_embed.proj, new_in_features=new_in_features)
        transformer.register_to_config(in_channels=new_in_features)

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {"transformer": transformer, "scheduler": scheduler}

    @torch.no_grad()
    def prepare_latents(
        self,
        vae: AutoencoderKL,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        control_image: Optional[torch.Tensor] = None,
        control_video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        _original_height: Optional[int] = None,
        _original_width: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        common_kwargs = {
            "vae": vae,
            "generator": generator,
            "compute_posterior": compute_posterior,
            "_original_height": _original_height,
            "_original_width": _original_width,
            **kwargs,
        }
        conditions = {"image": image, "video": video, **common_kwargs, **kwargs}
        input_keys = set(conditions.keys())
        conditions = super().prepare_latents(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}

        control_conditions = {"image": control_image, "video": control_video, **common_kwargs, **kwargs}
        input_keys = set(control_conditions.keys())
        control_conditions = super().prepare_latents(**control_conditions)
        control_conditions = {k: v for k, v in control_conditions.items() if k not in input_keys}

        return conditions, control_conditions

    def forward(
        self,
        transformer: CogView4Transformer2DModel,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        control_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        base_image_sequence_length = 256
        base_shift = 0.25
        max_shift = 0.75

        if compute_posterior:
            latents = latent_model_conditions.pop("latents")
            control_latents = control_model_conditions.pop("latents")
        else:
            posterior = DiagonalGaussianDistribution(latent_model_conditions.pop("latents"))
            latents = posterior.sample(generator=generator)
            del posterior

            control_posterior = DiagonalGaussianDistribution(control_model_conditions.pop("latents"))
            control_latents = control_posterior.sample(generator=generator)
            del control_posterior

        latents = (latents - self.vae_config.shift_factor) * self.vae_config.scaling_factor
        control_latents = (control_latents - self.vae_config.shift_factor) * self.vae_config.scaling_factor
        noise = torch.zeros_like(latents).normal_(generator=generator)
        timesteps = (sigmas.flatten() * 1000.0).long()

        image_sequence_length = latents.size(2) * latents.size(3) // self.transformer_config.patch_size**2
        mu = (image_sequence_length / base_image_sequence_length) ** 0.5
        mu = mu * max_shift + base_shift
        shifted_sigmas = mu / (mu + (1 / sigmas - 1) ** 1.0)
        noisy_latents = FF.flow_match_xt(latents, noise, shifted_sigmas)
        noisy_latents = torch.cat([noisy_latents, control_latents], dim=1)

        latent_model_conditions["hidden_states"] = noisy_latents.to(latents)

        pred = transformer(
            **latent_model_conditions,
            **condition_model_conditions,
            timestep=timesteps,
            return_dict=False,
        )[0]
        target = FF.flow_match_target(noise, latents)

        # NOTE: shifted_sigmas loss weighting seems to work better than sigmas. Needs more investigation
        # but let's keep it this way for now. Longer training runs should reveal more insights.
        # return pred, target, sigmas
        return pred, target, shifted_sigmas

    def _save_lora_weights(
        self,
        directory: str,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        norm_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
        *args,
        **kwargs,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            CogView4Pipeline.save_lora_weights(directory, transformer_state_dict, safe_serialization=True)
        if norm_state_dict is not None:
            safetensors.torch.save_file(norm_state_dict, os.path.join(directory, "norm_state_dict.safetensors"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    @property
    def _original_in_features(self):
        return self.transformer_config.in_channels

    @property
    def _control_layer_pattern(self):
        return "patch_embed.proj"

    @property
    def _qk_norm_identifiers(self):
        return ["attn1.norm_q", "attn1.norm_k"]
