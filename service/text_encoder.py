"""Kimodo's LLM2Vec representation loaded from the public Matbee NF4 export."""
from pathlib import Path
from unittest.mock import patch


class MatbeeEncoder:
    def __init__(self, model_path):
        import torch
        import transformers.modeling_utils as modeling_utils
        from kimodo.model.llm2vec.llm2vec import LLM2Vec

        root = Path(model_path)
        if not (root / 'config.json').is_file() or not (root / 'supervised_adapter/adapter_config.json').is_file():
            raise RuntimeError('Pinned Matbee export is incomplete; refusing a gated-model fallback')
        self._device = 'cuda:0'
        # Transformers 5.1's optional allocator warmup cannot traverse the NF4
        # buffer names. Skip only that optimization, only during this load.
        with patch.object(modeling_utils, 'caching_allocator_warmup', lambda *a, **kw: None):
            self.model = LLM2Vec.from_pretrained(
                base_model_name_or_path=str(root),
                peft_model_name_or_path=str(root / 'supervised_adapter'),
                torch_dtype=torch.bfloat16, device_map={'': self._device},
                low_cpu_mem_usage=True, local_files_only=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def to(self, device=None, dtype=None):
        # NF4 weights are placed by device_map and must not be dtype-cast later.
        if device is not None and not str(device).startswith('cuda'):
            raise ValueError('Matbee NF4 encoder requires CUDA')
        return self

    def eval(self):
        return self

    def get_device(self):
        return self._device

    def __call__(self, texts):
        import numpy as np
        import torch
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        with torch.inference_mode():
            encoded = self.model.encode(texts, batch_size=1, show_progress_bar=False, device=self._device)
        if encoded.ndim != 2 or encoded.shape[-1] != 4096 or not torch.isfinite(encoded).all():
            raise RuntimeError('Invalid LLM2Vec embedding')
        encoded = encoded[:, None].to(self._device)
        lengths = np.ones(len(encoded), dtype=int).tolist()
        return (encoded[0], lengths[0]) if single else (encoded, lengths)
