#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch

import lerobot.policies.smolvla.modeling_smolvla as modeling_smolvla
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


@pytest.mark.parametrize("resize", [(4, 4), None])
def test_prepare_images_normalizes_uint8_inputs_before_resizing(resize):
    policy = SimpleNamespace(
        config=SimpleNamespace(
            image_features={"observation.images.camera": object()},
            resize_imgs_with_padding=resize,
            empty_cameras=0,
        )
    )
    image = torch.tensor([[[[0, 255], [128, 64]]]], dtype=torch.uint8).repeat(1, 3, 1, 1)

    images, image_masks = SmolVLAPolicy.prepare_images(policy, {"observation.images.camera": image})

    assert images[0].dtype == torch.float32
    assert images[0].min() >= -1.0
    assert images[0].max() <= 1.0
    assert image_masks[0].tolist() == [True]
    if resize is None:
        torch.testing.assert_close(images[0], image.float() / 255.0 * 2.0 - 1.0)


@pytest.mark.parametrize(
    "grid",
    [(), (1.0, 0.0), (0.9, 0.5, 0.0), (1.0, 0.5, 0.1), (1.0, 0.6, 0.6, 0.0)],
)
def test_smolvla_rejects_invalid_denoise_time_grid(grid):
    with pytest.raises(ValueError, match="denoise_time_grid"):
        SmolVLAConfig(num_steps=3, denoise_time_grid=grid)


@pytest.mark.parametrize("grid", [None, (1.0, 0.25, 0.0)])
def test_smolvla_passes_time_grid_to_existing_integrator(monkeypatch, grid):
    seen = {}

    def fake_integrate(denoise_fn, noise, num_steps, **kwargs):
        seen["num_steps"] = num_steps
        seen["grid"] = kwargs["time_grid"]
        return noise

    monkeypatch.setattr(modeling_smolvla, "euler_integrate", fake_integrate)
    policy = SimpleNamespace(
        config=SimpleNamespace(
            chunk_size=2,
            max_action_dim=1,
            use_cache=False,
            num_steps=2,
            denoise_time_grid=grid,
        ),
        sample_noise=lambda shape, device: torch.zeros(shape, device=device),
        embed_prefix=lambda *args, **kwargs: (
            torch.zeros(1, 1, 1),
            torch.ones(1, 1, dtype=torch.bool),
            torch.zeros(1, 1, dtype=torch.bool),
        ),
        vlm_with_expert=SimpleNamespace(forward=lambda **kwargs: (None, None)),
        _rtc_enabled=lambda: False,
        rtc_processor=None,
    )
    state = torch.zeros(1, 1)

    modeling_smolvla.VLAFlowMatching.sample_actions(policy, [], [], None, None, state)

    assert seen["num_steps"] == 2
    if grid is None:
        assert seen["grid"] is None
    else:
        torch.testing.assert_close(seen["grid"], torch.tensor(grid))
