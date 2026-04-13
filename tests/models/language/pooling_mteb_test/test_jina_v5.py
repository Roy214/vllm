# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import partial

import pytest

from tests.models.language.pooling.embed_utils import (
    correctness_test_embed_models,
)
from tests.models.utils import EmbedModelInfo

from .mteb_embed_utils import mteb_test_embed_models

# jina-embeddings-v5-text-small is Qwen3-0.6B-Base + four task-specific
# LoRA adapters. vLLM merges the selected adapter at load time; pick the
# task via --hf-overrides '{"jina_task": "..."}'. Default is "retrieval".
EMBEDDING_MODELS = [
    EmbedModelInfo(
        "jinaai/jina-embeddings-v5-text-small",
        architecture="JinaEmbeddingsV5Model",
        seq_pooling_type="LAST",
        trust_remote_code=True,
    )
]


@pytest.mark.parametrize("model_info", EMBEDDING_MODELS)
def test_embed_models_mteb(
    hf_runner, vllm_runner, model_info: EmbedModelInfo
) -> None:
    # Match the adapter task used by vLLM (defaults to "retrieval").
    def hf_model_callback(model):
        model.encode = partial(model.encode, task="retrieval.query")

    mteb_test_embed_models(
        hf_runner, vllm_runner, model_info, hf_model_callback=hf_model_callback
    )


@pytest.mark.parametrize("model_info", EMBEDDING_MODELS)
def test_embed_models_correctness(
    hf_runner,
    vllm_runner,
    model_info: EmbedModelInfo,
    example_prompts,
) -> None:
    def hf_model_callback(model):
        model.encode = partial(model.encode, task="retrieval.query")

    correctness_test_embed_models(
        hf_runner,
        vllm_runner,
        model_info,
        example_prompts,
        hf_model_callback=hf_model_callback,
    )
