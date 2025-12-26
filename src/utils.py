import torch


class KVCache:

    def __init__(self, max_cache_len=None, sliding_window=None, offload_to_cpu=False):
   
        self.keys = []   
        self.values = []  
        self.sliding_window = sliding_window
        self.max_cache_len = max_cache_len
        self.offload_to_cpu = offload_to_cpu
        self.seen_tokens = 0

    def get_seq_length(self, layer_idx=0):
        if layer_idx >= len(self.keys) or self.keys[layer_idx] is None:
            return 0
        return self.keys[layer_idx].shape[-2]  

    def update(self, key_states, value_states, layer_idx):

        while layer_idx >= len(self.keys):
            self.keys.append(None)
            self.values.append(None)

        cached_keys = self.keys[layer_idx]
        cached_values = self.values[layer_idx]

        if cached_keys is None:
            keys = key_states
            values = value_states
        else:
            if cached_keys.device != key_states.device:
                cached_keys = cached_keys.to(key_states.device, non_blocking=True)
                cached_values = cached_values.to(key_states.device, non_blocking=True)
            keys = torch.cat([cached_keys, key_states], dim=-2)
            values = torch.cat([cached_values, value_states], dim=-2)

        if self.sliding_window is not None:
            window_size = max(self.sliding_window - 1, 0)
            keys = keys[..., -window_size:, :] if window_size > 0 else keys[..., :0, :]
            values = values[..., -window_size:, :] if window_size > 0 else values[..., :0, :]
        
        elif self.max_cache_len is not None and keys.shape[-2] > self.max_cache_len:
            keys = keys[..., -self.max_cache_len:, :]
            values = values[..., -self.max_cache_len:, :]

        if self.offload_to_cpu:
            self.keys[layer_idx] = keys.to("cpu", non_blocking=True)
            self.values[layer_idx] = values.to("cpu", non_blocking=True)
        else:
            self.keys[layer_idx] = keys
            self.values[layer_idx] = values

        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        return keys, values


