"""Profile the second, steady-state LiveAct block without changing repo code.

Run from the SoulX-LiveAct checkout after preparing /tmp/liveact-profile-224x384-1p5s.json.
The JSON should contain one item with a 1.5-second audio excerpt, so the
generator performs exactly two blocks. Results are written under /tmp.
"""

import functools
import json
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path.cwd()))

from fp8_gemm import FP8Linear
from model_liveact import model_memory


OUTPUT = Path('/tmp/liveact-profile-fresh-pageable-224x384')
OUTPUT.mkdir(exist_ok=True)
prof = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
)
state = {'calls': 0, 'started': False, 'finished': False, 'wall_s': None}


def label_method(owner, name, label):
    original = getattr(owner, name)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        with record_function(label):
            return original(*args, **kwargs)

    setattr(owner, name, wrapped)


label_method(FP8Linear, 'forward', 'liveact/fp8_linear')
label_method(model_memory.WanBlockOffloadManager, '_copy_tensor', 'liveact/weight_copy')

original_move_kv = model_memory.WanSelfAttention._move_kv_cache_to_device


@functools.wraps(original_move_kv)
def move_kv(self, kv_cache, device):
    direction = 'kv_d2h' if torch.device(device).type == 'cpu' else 'kv_h2d'
    with record_function(f'liveact/{direction}'):
        return original_move_kv(self, kv_cache, device)


model_memory.WanSelfAttention._move_kv_cache_to_device = move_kv

original_sdpa = model_memory.sdpa_attention


@functools.wraps(original_sdpa)
def sdpa(*args, **kwargs):
    with record_function('liveact/sdpa_attention'):
        return original_sdpa(*args, **kwargs)


model_memory.sdpa_attention = sdpa

original_forward = model_memory.WanModel.forward


@functools.wraps(original_forward)
def model_forward(self, *args, **kwargs):
    call_index = state['calls']
    state['calls'] += 1
    if call_index == 3:
        torch.cuda.synchronize()
        prof.start()
        state['started'] = True
        state['start_time'] = time.perf_counter()

    with record_function('liveact/dit_forward'):
        result = original_forward(self, *args, **kwargs)

    if call_index == 3:
        torch.cuda.synchronize()
        state['wall_s'] = time.perf_counter() - state['start_time']
        prof.stop()
        state['finished'] = True
        trace = OUTPUT / 'steady_block_trace.json'
        prof.export_chrome_trace(str(trace))
        events = []
        for event in prof.key_averages():
            if event.key.startswith('liveact/'):
                events.append({
                    'name': event.key,
                    'count': event.count,
                    'cpu_total_ms': round(event.cpu_time_total / 1000, 3),
                    'device_total_ms': round(getattr(event, 'device_time_total', 0) / 1000, 3),
                })
        report = {
            'profiled_forwards': 1,
            'steady_block_dit_wall_s': round(state['wall_s'], 3),
            'events': events,
            'trace': str(trace),
        }
        (OUTPUT / 'summary.json').write_text(json.dumps(report, indent=2))
        (OUTPUT / 'operator_table.txt').write_text(
            prof.key_averages().table(sort_by='device_time_total', row_limit=50)
        )
        print('LIVEACT_PROFILE_SUMMARY', json.dumps(report), flush=True)

    return result


model_memory.WanModel.forward = model_forward

from generate import _parse_args, generate  # noqa: E402

sys.argv = [
    'generate.py',
    '--size', '224*384',
    '--ckpt_dir', 'checkpoints/LiveAct',
    '--wav2vec_dir', 'checkpoints/chinese-wav2vec2-base',
    '--fps', '24',
    '--input_json', '/tmp/liveact-profile-fresh-pageable-224x384-1p5s.json',
    '--fp8_cache_dir', '/tmp/liveact-fp8-cache',
    '--prompt_cache_dir', '/tmp/liveact-prompt-cache',
    '--resident_kv_steps', '1',
    '--fp8_gemm', '--fp8_kv_cache', '--offload_cache',
    '--block_offload', '--t5_cpu', '--disable_compile', '--dura_print',
]
generate(_parse_args())
if not state['finished']:
    raise RuntimeError(f"Expected four DiT forwards, got {state['calls']}")
