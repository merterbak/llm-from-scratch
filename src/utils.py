import torch


class DynamicCache:

    def __init__(self, max_cache_len=None, sliding_window=None, offload_to_cpu=False):
        self.key_cache = []
        self.value_cache = []
        if max_cache_len is None and sliding_window is not None:
            max_cache_len = sliding_window
        self.max_cache_len = int(max_cache_len) if max_cache_len is not None else None

        self.offload_to_cpu = bool(offload_to_cpu)

        self.seen_tokens = 0

    def get_seq_length(self, layer_idx=0):
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            return 0
        return int(self.key_cache[layer_idx].size(2))

    def update(self, key_states, value_states, layer_idx):
        compute_device = key_states.device

        if layer_idx >= len(self.key_cache):
            grow = layer_idx - len(self.key_cache) + 1
            self.key_cache.extend([None] * grow)
            self.value_cache.extend([None] * grow)

        if self.offload_to_cpu:
            key_states = key_states.to("cpu")
            value_states = value_states.to("cpu")

        if self.key_cache[layer_idx] is None:
            k = key_states
            v = value_states
        else:
            k = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            v = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        if self.max_cache_len is not None and k.size(2) > self.max_cache_len:
            k = k[:, :, -self.max_cache_len :, :]
            v = v[:, :, -self.max_cache_len :, :]

        self.key_cache[layer_idx] = k
        self.value_cache[layer_idx] = v

        if layer_idx == 0:
            self.seen_tokens += int(key_states.size(2))

        if self.offload_to_cpu:
            return k.to(compute_device), v.to(compute_device)
        return k, v


