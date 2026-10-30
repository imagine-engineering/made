"""Execute unchanged pinned VAE nodes, wrapper methods, tiler and PixelSpace model."""
import argparse
import ast
import contextlib
import copy
import hashlib
import itertools
import json
import logging
import math
from pathlib import Path
import time
import types

import torch
from safetensors.torch import save

ROOT = Path('/home/asvegah/repos/sim/projects/sim/ComfyUI')
DESTINATION = Path(__file__).with_name('source-contract.json')
FILES = ['nodes.py', 'sim/sd.py', 'sim/utils.py', 'sim/pixel_space_convert.py',
         'sim_extras/nodes_audio.py', 'sim_extras/nodes_triposplat.py', 'sim_extras/nodes_void.py', 'execution.py']
CLAIMS = {'VAEDecodeTiled': 'COMFY-NODE-0714', 'VAEDecodeTripoSplat': 'COMFY-NODE-0715',
          'VAEEncode': 'COMFY-NODE-0716', 'VAEEncodeAudio': 'COMFY-NODE-0717',
          'VAEEncodeForInpaint': 'COMFY-NODE-0718', 'VAEEncodeTiled': 'COMFY-NODE-0719',
          'VOIDWarpedNoise': 'COMFY-NODE-0744', 'VOIDWarpedNoiseSource': 'COMFY-NODE-0745'}
EXECUTABLE = ['VAEDecodeTiled', 'VAEEncode', 'VAEEncodeForInpaint', 'VAEEncodeTiled']

def select(path, names, namespace):
    selected = [copy.deepcopy(node) for node in ast.parse((ROOT / path).read_text()).body
                if getattr(node, 'name', None) in names]
    assert len(selected) == len(names), (path, names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 str(ROOT / path), 'exec'), namespace)


def wire(tensor):
    if tensor is None:
        return None
    values = tensor.detach().reshape(-1).tolist()
    bits = tensor.detach().contiguous().view(torch.int32).reshape(-1).tolist()
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'values': [value if math.isfinite(value) else {'nonfinite_f32_bits': raw & 0xffffffff}
                       for value, raw in zip(values, bits)]}



def original_noise_source_cases():
    io = types.SimpleNamespace(ComfyNode=object, NodeOutput=lambda *values: values,
        Schema=lambda **options: options,
        Latent=types.SimpleNamespace(Input=lambda name, **options: {'name': name, 'type': 'LATENT', **options}),
        Noise=types.SimpleNamespace(Output=lambda **options: {'type': 'NOISE', **options}))
    namespace = {'torch': torch, 'io': io}
    select('sim_extras/nodes_void.py', {'Noise_FromLatent', 'VOIDWarpedNoiseSource'}, namespace)
    node = namespace['VOIDWarpedNoiseSource']
    entries = []
    def observe(samples, supplied, label, capability=None):
        before = samples.clone() if isinstance(samples, torch.Tensor) else None
        item = {'node': 'VOIDWarpedNoiseSource', 'label': label, 'supplied_input': supplied}
        if isinstance(samples, torch.Tensor):
            item['samples'] = {'shape': list(samples.shape), 'dtype': str(samples.dtype),
                               'values': samples.reshape(-1).tolist(), 'strides': list(samples.stride())}
        else:
            item['samples'] = {'nested_source_dictionary': True}
        if capability is not None:
            item['capability_gap'] = capability
        input_value = {'none': None, 'object': object(), 'scalar': torch.tensor(99.),
                       'different-shape': torch.zeros(2, 5, 1)}[supplied]
        try:
            source = node.execute({'samples': samples, 'batch_index': [17], 'ignored_source_metadata': True})[0]
            first = source.generate_noise(input_value)
            repeated = source.generate_noise(input_value)
            item['seed'] = source.seed
            item['output'] = {'shape': list(first.shape), 'dtype': str(first.dtype), 'values': first.reshape(-1).tolist()}
            item['output_device'] = str(first.device)
            item['independent_storage'] = first.numel() == 0 or (first.data_ptr() != samples.data_ptr()
                                                               and repeated.data_ptr() != first.data_ptr())
            if first.numel():
                first[tuple(0 for _ in first.shape)] = 123.
            item['repeat_unchanged_after_output_mutation'] = torch.equal(repeated, before)
        except Exception as error:
            item['error_type'] = type(error).__name__
            item['error'] = str(error)
        if before is not None:
            item['samples_unchanged'] = torch.equal(samples, before)
        entries.append(item)
    for shape in [(1, 1, 2, 2), (2, 3, 2, 3), (1, 2, 3, 2, 2), (1, 2, 3)]:
        samples = ((torch.arange(math.prod(shape), dtype=torch.float32) % 7 - 3) / 8).reshape(shape)
        for supplied in ['none', 'object', 'scalar', 'different-shape']:
            observe(samples, supplied, f'{shape}; input {supplied}')
    for samples, label, gap in [
            (torch.tensor([[.25, -.5]]), 'rank-two samples', 'canonical LATENT ranks3/4/5 only'),
            (torch.zeros(0, 1, 2, 2), 'zero batch samples', 'positive canonical noise geometry required'),
            (torch.tensor([[[[.25, -.5]]]], dtype=torch.float64), 'float64 samples', 'canonical noise F32 only'),
            (torch.arange(12., dtype=torch.float32).reshape(1, 1, 3, 4).transpose(-1, -2),
             'strided retained samples', 'canonical noise contiguous samples only')]:
        observe(samples, 'none', label, gap)
    observe({'video': torch.zeros(1, 1, 1, 1, 1), 'audio': torch.zeros(1, 1, 1)},
            'none', 'nested AV samples has no clone method')
    return node.define_schema(), entries


def generate():
    torch.set_num_threads(1)
    utility = {'torch': torch, 'itertools': itertools, 'math': math, 'time': time,
               'PROGRESS_BAR_HOOK': None, 'PROGRESS_THROTTLE_MIN_INTERVAL': 0.1,
               'PROGRESS_THROTTLE_MIN_PERCENT': 0.5}
    select('sim/utils.py', {'get_tiled_scale_steps', 'tiled_scale_multidim', 'tiled_scale', 'ProgressBar'}, utility)
    model_namespace = {'torch': torch}
    select('sim/pixel_space_convert.py', {'PixelspaceConversionVAE'}, model_namespace)
    model = model_namespace['PixelspaceConversionVAE']().eval()
    def raise_non_oom(error):
        raise error
    management = types.SimpleNamespace(intermediate_dtype=lambda: torch.float32,
        cuda_device_context=lambda device: contextlib.nullcontext(), load_models_gpu=lambda *args, **kwargs: None,
        raise_non_oom=raise_non_oom, dtype_size=lambda dtype: torch.empty((), dtype=dtype).element_size())
    sim = types.SimpleNamespace(utils=types.SimpleNamespace(**utility), model_management=management,
                                 pixel_space_convert=types.SimpleNamespace(**model_namespace))
    original = next(node for node in ast.parse((ROOT / 'sim/sd.py').read_text()).body if getattr(node, 'name', '') == 'VAE')
    methods = {'encode', 'encode_tiled', 'encode_tiled_', 'decode_tiled', 'decode_tiled_',
               'throw_exception_if_invalid', 'vae_encode_crop_pixels', 'vae_output_dtype',
               'spacial_compression_encode', 'spacial_compression_decode', 'temporal_compression_decode'}
    members = [copy.deepcopy(node) for node in original.body if getattr(node, 'name', '') in methods]
    assert len(members) == len(methods)
    wrapper = ast.ClassDef(name='OriginalImageVAE', bases=[], keywords=[], body=members, decorator_list=[])
    wrappers = {'torch': torch, 'sim': sim, 'model_management': management, 'logging': logging}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(ROOT / 'sim/sd.py'), 'exec'), wrappers)
    resource = wrappers['OriginalImageVAE']()
    constructor = next(node for node in original.body if getattr(node, 'name', '') == '__init__')
    policies = {'process_input', 'process_output', 'crop_input', 'pad_channel_value',
                'not_video', 'extra_1d_channel', 'downscale_index_formula', 'upscale_index_formula'}
    assignments = [copy.deepcopy(node) for node in constructor.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Attribute) and target.attr in policies for target in node.targets)]
    assert len(assignments) == len(policies)
    pixel_branch = next(node for node in ast.walk(constructor) if isinstance(node, ast.If)
                        and ast.unparse(node.test) == "'pixel_space_vae' in sd")
    exec(compile(ast.fix_missing_locations(ast.Module(body=assignments + copy.deepcopy(pixel_branch.body), type_ignores=[])), str(ROOT / 'sim/sd.py'), 'exec'),
         {'self': resource, 'sim': sim, 'model_management': management})
    resource.first_stage_model = model
    resource.device = resource.output_device = torch.device('cpu')
    resource.vae_dtype = torch.float32
    resource.patcher = types.SimpleNamespace(get_free_memory=lambda device: 1024 * 1024 * 1024)
    nodes = {'torch': torch, 'math': math}
    select('nodes.py', set(EXECUTABLE), nodes)
    entries = []
    def observe(node, shape, offset, options, label):
        count = math.prod(shape)
        value = (torch.arange(count, dtype=torch.float32) % 11 - 3) / 8 + offset
        value = value.reshape(shape)
        before = value.clone()
        item = {'node': node, 'label': label, 'input': wire(value), 'options': options}
        try:
            if node == 'VAEDecodeTiled':
                output = nodes[node]().decode(resource, {'samples': value, 'batch_index': [17], 'ignored_source_metadata': True}, **options)[0]
            else:
                output = nodes[node]().encode(resource, value, **options)[0]['samples']
            item['output'] = wire(output)
        except (NameError, ImportError, AttributeError):
            raise
        except Exception as error:
            item['error_type'] = type(error).__name__
            item['error'] = str(error)
        item['input_unchanged'] = torch.equal(value, before)
        entries.append(item)
    for shape, offset, label in [((1, 2, 3, 3), 0., 'non-square RGB'), ((2, 3, 5, 3), .25, 'two images'),
                                 ((1, 2, 3, 4), 0., 'source truncates alpha'), ((1, 2, 3, 3), 2., 'source encode no clamp'),
                                 ((1, 2, 3, 1), 0., 'source grayscale channel representation boundary'),
                                 ((0, 2, 3, 3), 0., 'empty batch body boundary')]:
        observe('VAEEncode', shape, offset, {}, label)
    for shape, label in [((1, 2, 3, 3), 'single tile'), ((2, 3, 5, 3), 'batch tiles'),
                         ((1, 65, 2, 3), 'vertical logical edge tiles'), ((1, 2, 65, 3), 'horizontal logical edge tiles')]:
        for overlap in [0, 8, 16]:
            observe('VAEEncodeTiled', shape, 0., {'tile_size': 64, 'overlap': overlap, 'temporal_size': 64, 'temporal_overlap': 8}, f'{label}; overlap{overlap}')
    for overlap in [32, 64, 128]:
        observe('VAEEncodeTiled', (1, 65, 2, 3), 0., {'tile_size': 64, 'overlap': overlap}, f'source tiler invalid step overlap{overlap}')
    for shape, offset, label in [((1, 3, 2, 3), 0., 'single latent tile'), ((2, 3, 3, 5), .25, 'two latent images'),
                                 ((1, 3, 65, 2), 0., 'vertical logical edge tiles'), ((1, 3, 2, 65), 0., 'horizontal logical edge tiles'),
                                 ((1, 3, 2, 3), 2., 'source output upper clamp')]:
        for overlap in [0, 8, 20, 64]:
            observe('VAEDecodeTiled', shape, offset, {'tile_size': 64, 'overlap': overlap, 'temporal_size': 8, 'temporal_overlap': 64}, f'{label}; overlap{overlap}')
    inpaint_entries = []
    preparation_entries = []
    class CapturePreparationVae:
        def __init__(self, compression):
            self.compression = compression
            self.pixels = None
        def spacial_compression_encode(self):
            return self.compression
        def encode(self, pixels):
            self.pixels = pixels.clone()
            return None
    def observe_inpaint(shape, mask_shape, grow, label, compression=None):
        pixels = ((torch.arange(math.prod(shape), dtype=torch.float32) % 11 - 3) / 8).reshape(shape)
        mask_values = torch.tensor([-1., 0., .25, .5, .75, 1., 1.5, 2.], dtype=torch.float32)
        mask = mask_values[torch.arange(math.prod(mask_shape)) % mask_values.numel()].reshape(mask_shape)
        before_pixels, before_mask = pixels.clone(), mask.clone()
        item = {'node': 'VAEEncodeForInpaint', 'label': label, 'input': wire(pixels),
                'mask': wire(mask), 'options': {'grow_mask_by': grow}}
        owner = resource if compression is None else CapturePreparationVae(compression)
        if compression is not None:
            item['compression'] = compression
            item['scope'] = 'original node body; capture-only VAE encode boundary; preparation proof only, no model output'
        try:
            result = nodes['VAEEncodeForInpaint']().encode(owner, pixels, mask, grow)[0]
            if compression is None:
                item['output'] = wire(result['samples'])
            else:
                assert result['samples'] is None
                item['prepared_pixels'] = wire(owner.pixels)
            item['noise_mask'] = wire(result['noise_mask'])
        except (NameError, ImportError, AttributeError):
            raise
        except Exception as error:
            item['error_type'] = type(error).__name__
            item['error'] = str(error)
        item['input_unchanged'] = torch.equal(pixels, before_pixels)
        item['mask_unchanged'] = torch.equal(mask, before_mask)
        (inpaint_entries if compression is None else preparation_entries).append(item)
    for grow in [0, 1, 2, 3, 6, 64]:
        observe_inpaint((1, 3, 5, 3), (1, 2, 3), grow, f'bilinear resize and ties-even; grow{grow}')
    for shape, mask_shape, label in [
            ((2, 3, 5, 3), (1, 3, 5), 'one mask retained for two images'),
            ((2, 3, 5, 3), (2, 3, 5), 'independent two-mask batch'),
            ((1, 3, 5, 3), (2, 3, 5), 'in-place broadcast cannot enlarge image batch'),
            ((2, 3, 5, 3), (3, 3, 5), 'incompatible source mask batch'),
            ((1, 3, 5, 4), (1, 3, 5), 'source alpha unchanged before VAE truncation'),
            ((1, 3, 5, 3), (3, 5), 'source unbatched mask representation boundary'),
            ((1, 3, 5, 1), (1, 3, 5), 'source fewer-than-three-channel failure')]:
        for grow in [0, 2]:
            observe_inpaint(shape, mask_shape, grow, f'{label}; grow{grow}')
    for compression in [2, 3, 8]:
        for grow in [0, 2, 3, 6]:
            observe_inpaint((2, compression + 3, compression * 2 + 3, 4),
                            (1, 2, 3), grow, f'centered crop ratio{compression}; grow{grow}', compression)
    noise_schema, noise_entries = original_noise_source_cases()
    artifact = save(model.state_dict())
    return {'claims': CLAIMS, 'executable_nodes': EXECUTABLE,
            'schemas': {name: nodes[name].INPUT_TYPES() for name in EXECUTABLE},
            'aliases': {'VAEEncode': nodes['VAEEncode'].SEARCH_ALIASES},
            'cases': entries,
            'inpaint_cases': inpaint_entries, 'inpaint_preparation_cases': preparation_entries,
            'noise_source_schema': noise_schema, 'noise_source_cases': noise_entries,
            'noise_source_scope': 'unchanged original Noise_FromLatent and VOIDWarpedNoiseSource AST; original V3 define_schema through capture-only IO descriptors; genuine CPU clone values/storage/repeated-mutation behavior; no native pixel or tensor value determines source golds',
            'source_sha256': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in FILES},
            'artifact': {'bytes': list(artifact), 'sha256': hashlib.sha256(artifact).hexdigest(),
                         'producer': 'complete original PixelspaceConversionVAE state_dict, safetensors0.4.2'},
            'source_environment': {'torch': torch.__version__, 'threads': torch.get_num_threads(), 'device': 'cpu', 'dtype': 'torch.float32'},
            'scope': 'original PromptExecutor inference_mode execution context; unchanged original node class bodies, selected complete original VAE wrapper methods, actual original PixelSpace model and original tiler; no native output determines fixture values',
            'infrastructure': 'CPU residency/free-memory callbacks controlled; original progress class with absent UI hook; no OOM fallback, video, actual clock or native app proof',
            'remaining_gates': ['full learned and video VAE resources', 'VAEEncodeForInpaint non-F32/device/arbitrary-rank mask boundaries', 'VAEEncodeAudio sample-rate/axis owner',
                                'TripoSplat seeded decoder request', 'VOIDWarpedNoise RAFT/warp/RNG owner', 'VOIDWarpedNoiseSource clone/ignore-input owner']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    arguments = parser.parse_args()
    with torch.inference_mode():
        value = generate()
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if arguments.check:
        if DESTINATION.read_text() != encoded:
            raise RuntimeError('Immutable source contract differs')
    else:
        DESTINATION.write_text(encoded)
    print(f"Original latent noise cases{len(value['noise_source_cases'])}; successes{sum('output' in case for case in value['noise_source_cases'])}; errors{sum('error_type' in case for case in value['noise_source_cases'])}")
    print(f"Inpaint real PixelSpace cases{len(value['inpaint_cases'])}; preparation-only cases{len(value['inpaint_preparation_cases'])}")
    print(f"Original source cases{len(value['cases'])}; successes{sum('output' in case for case in value['cases'])}; errors{sum('error_type' in case for case in value['cases'])}")

if __name__ == '__main__':
    main()
