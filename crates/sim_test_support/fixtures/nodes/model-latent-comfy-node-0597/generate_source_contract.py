import argparse
import asyncio
import typing
import ast
import hashlib
import json
import math
import types
from pathlib import Path

import torch

ROOT = Path('/home/asvegah/repos/sim/projects/sim/ComfyUI')
FILES = ['nodes.py', 'sim_extras/nodes_wan.py', 'sim_extras/nodes_mask.py', 'execution.py', 'sim_extras/nodes_stable_cascade.py', 'sim_execution/graph.py']
torch.set_num_threads(1)


def select(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    body = [node for node in tree.body if getattr(node, 'name', None) in names]
    assert len(body) == len(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(ROOT / path), 'exec'), namespace)


io = types.SimpleNamespace(ComfyNode=object, NodeOutput=lambda *values: values)
sim = types.SimpleNamespace(model_management=types.SimpleNamespace(
    intermediate_device=lambda: torch.device('cpu'), intermediate_dtype=lambda: torch.float32))
namespace = {'torch': torch, 'io': io, 'IO': io, 'sim': sim, 'MAX_RESOLUTION': 16384,
             'nodes': types.SimpleNamespace(MAX_RESOLUTION=16384)}
select('nodes.py', ['SetLatentNoiseMask', 'EmptyLatentImage'], namespace)
select('sim_extras/nodes_wan.py', ['TrimVideoLatent'], namespace)
select('sim_extras/nodes_mask.py', ['SolidMask'], namespace)
select('sim_extras/nodes_stable_cascade.py', ['StableCascade_EmptyLatentImage'], namespace)
execution = ast.parse((ROOT / 'execution.py').read_text())
mapper = next(node for node in execution.body if getattr(node, 'name', '') == '_async_map_node_over_list')
slice_dictionary = next(node for node in mapper.body if getattr(node, 'name', '') == 'slice_dict')
exec(compile(ast.Module(body=[slice_dictionary], type_ignores=[]), str(ROOT / 'execution.py'), 'exec'), namespace)


def tensor(shape, offset=0):
    return (torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape).remainder(17) - 8) * 0.125 + offset


def wire_tensor(value):
    return {'shape': list(value.shape), 'values': value.flatten().tolist()}


def wire(value):
    return {key: wire_tensor(item) if isinstance(item, torch.Tensor) else item for key, item in value.items()}


def latent(shape, mask_shape=None, indices=None):
    value = {'samples': tensor(shape), 'downscale_ratio_spacial': 8, 'downscale_ratio_temporal': 4,
             'source_marker': 'original dictionary key retained'}
    if mask_shape is not None:
        value['noise_mask'] = tensor(mask_shape, 0.25)
    if indices is not None:
        value['batch_index'] = indices
    return value


cases = []
gaps = []


def record(name, samples, argument, label='', destination=cases):
    value = {'node': name, 'input': wire(samples), 'label': label}
    if name == 'SetLatentNoiseMask':
        value['mask'] = wire_tensor(argument)
    else:
        value['trim_amount'] = argument
    try:
        if name == 'SetLatentNoiseMask':
            result = namespace[name]().set_mask(samples, argument)[0]
        else:
            result = namespace[name].execute(samples, argument)[0]
        value['output'] = wire(result)
        value['returns_original_dictionary'] = result is samples
        value['samples_alias_input_storage'] = result['samples'].untyped_storage().data_ptr() == samples['samples'].untyped_storage().data_ptr()
    except Exception as error:
        value['error_type'] = type(error).__name__
        value['error'] = str(error)
    destination.append(value)


for shape in [(1, 2, 3, 5), (2, 2, 3, 5), (2, 2, 4, 3, 5)]:
    for mask_shape in [(1, 3, 5), (3, 3, 5), (2, 9, 7), (0, 3, 5), (2, 0, 5)]:
        for indices in [None, [], [99]]:
            record('SetLatentNoiseMask', latent(shape, (1, 1, 3, 5), indices), tensor(mask_shape))

for shape in [(2, 4, 5), (2, 4, 5, 3), (2, 4, 5, 3, 5)]:
    mask_shape = (1, 1, 5) if len(shape) == 3 else (1, 1, 3, 5)
    for amount in [0, 1, 4, 5, 99, -1, -99]:
        for indices in [None, [], [77, 88, 99]]:
            record('TrimVideoLatent', latent(shape, mask_shape, indices), amount)

for mask_shape in [(3, 5), (2, 1, 3, 5), (1, 2, 1, 3, 5)]:
    record('SetLatentNoiseMask', latent((2, 2, 3, 5)), tensor(mask_shape),
           'source-success/native MASK rank-three admission gap', gaps)
record('SetLatentNoiseMask', latent((2, 4, 5)), tensor((1, 3, 5)),
       'source-success/native audio-rank-three LATENT cannot retain reshaped rank-four mask', gaps)

errors = []
for name, samples, argument in [
    ('SetLatentNoiseMask', latent((1, 2, 3, 5)), torch.tensor([1.0])),
    ('SetLatentNoiseMask', latent((1, 2, 3, 5)), [1.0]),
    ('SetLatentNoiseMask', [], tensor((1, 3, 5))),
    ('TrimVideoLatent', {'samples': tensor((2, 3))}, 1),
    ('TrimVideoLatent', [], 1),
    ('TrimVideoLatent', {}, 1),
]:
    try:
        if name == 'SetLatentNoiseMask':
            namespace[name]().set_mask(samples, argument)
        else:
            namespace[name].execute(samples, argument)
        raise AssertionError('Expected original source failure')
    except Exception as error:
        errors.append({'node': name, 'error_type': type(error).__name__, 'error': str(error)})

mapped = []
for name in ['SetLatentNoiseMask', 'TrimVideoLatent']:
    inputs = {'samples': [namespace['EmptyLatentImage']().generate(64, 64, batch)[0] for batch in [1, 2, 3]]}
    if name == 'SetLatentNoiseMask':
        inputs['mask'] = [namespace['SolidMask'].execute(value, 8, 8)[0] for value in [0.25, 0.75]]
    else:
        inputs['trim_amount'] = [0, 1]
    outputs = []
    for index in range(max(map(len, inputs.values()))):
        current = namespace['slice_dict'](inputs, index)
        if name == 'SetLatentNoiseMask':
            output = namespace[name]().set_mask(**current)[0]
        else:
            output = namespace[name].execute(**current)[0]
        outputs.append(wire(output))
    mapped.append({'node': name, 'source_repeat_last': True, 'batch_sizes': [1, 2, 3],
                   'values': [0.25, 0.75] if name == 'SetLatentNoiseMask' else [0, 1], 'outputs': outputs})

cascade = []
cascade_gaps = []

def record_cascade(arguments, label, destination=cascade, dtype=torch.float32):
    previous = torch.get_default_dtype()
    value = {'node': 'StableCascade_EmptyLatentImage', 'arguments': arguments, 'label': label,
             'default_dtype': str(dtype)}
    try:
        torch.set_default_dtype(dtype)
        outputs = namespace['StableCascade_EmptyLatentImage'].execute(**arguments)
        value['outputs'] = [wire(output) for output in outputs]
        value['output_dtypes'] = [str(output['samples'].dtype) for output in outputs]
    except Exception as error:
        value['error_type'] = type(error).__name__
        value['error'] = str(error)
    finally:
        torch.set_default_dtype(previous)
    destination.append(value)

for arguments, label in [
    ({'width': 256, 'height': 256, 'compression': 42, 'batch_size': 1}, 'smallest schema geometry'),
    ({'width': 257, 'height': 265, 'compression': 42, 'batch_size': 2}, 'nondivisible asymmetric geometry'),
    ({'width': 256, 'height': 256, 'compression': 128, 'batch_size': 3}, 'high compression multiple batches'),
    ({'width': 256, 'height': 256, 'compression': 4, 'batch_size': 1}, 'minimum compression'),
    ({'width': 1024, 'height': 1024, 'compression': 42}, 'source schema defaults and omitted optional batch'),
]:
    record_cascade(arguments, label)
for arguments, label in [
    ({'width': 256, 'height': 256, 'compression': 0, 'batch_size': 1}, 'source zero division'),
    ({'width': -256, 'height': 256, 'compression': 42, 'batch_size': 1}, 'source negative dimension'),
    ({'width': 256, 'height': 256, 'compression': 42, 'batch_size': -1}, 'source negative batch'),
    ({'width': 256.5, 'height': 256, 'compression': 42, 'batch_size': 1}, 'source noninteger factory dimension'),
]:
    record_cascade(arguments, label)
for arguments, label in [
    ({'width': 0, 'height': 256, 'compression': 42, 'batch_size': 1}, 'source-success/schema width below256'),
    ({'width': 256, 'height': 0, 'compression': 42, 'batch_size': 1}, 'source-success/schema height below256'),
    ({'width': 256, 'height': 256, 'compression': 42, 'batch_size': 0}, 'source-success/schema batch below1'),
    ({'width': 2, 'height': 3, 'compression': 42, 'batch_size': 1}, 'source-success/schema zero-sized stages'),
    ({'width': 256, 'height': 256, 'compression': 1024, 'batch_size': 1}, 'source-success/schema compression above128 with zero stageC'),
]:
    record_cascade(arguments, label, cascade_gaps)
record_cascade({'width': 256, 'height': 256, 'compression': 42, 'batch_size': 1},
               'source-success/default float64 differs from canonical node float32', cascade_gaps, torch.float64)
# These labels reference unchanged original case objects rather than rewriting their golds.
schema_gaps = [{'case_index': index, 'node': case['node'], 'label': 'source-success/trim_amount below schema minimum0'}
               for index, case in enumerate(cases)
               if case['node'] == 'TrimVideoLatent' and case['trim_amount'] < 0]
io.Int = types.SimpleNamespace(Input=lambda name, **options: {'name': name, 'type': 'INT', **options})
io.Latent = types.SimpleNamespace(Input=lambda name, **options: {'name': name, 'type': 'LATENT', **options}, Output=lambda **options: options)
io.Schema = lambda **options: options
cascade_schema = namespace['StableCascade_EmptyLatentImage'].define_schema()
trim_schema = namespace['TrimVideoLatent'].define_schema()
io.Combo = types.SimpleNamespace(io_type='COMBO')
namespace.update({'_ComfyNodeInternal': type('UnrelatedV3Base', (), {}), 'Type': typing.Type,
                  'Literal': typing.Literal, 'ComfyNodeABC': object, 'InputTypeDict': dict,
                  'InputTypeOptions': dict})
select('sim_execution/graph.py', ['get_input_info'], namespace)
select('execution.py', ['validate_inputs'], namespace)
for name, schema in [('TrimVideoLatent', trim_schema), ('StableCascade_EmptyLatentImage', cascade_schema)]:
    required = {item['name']: (item['type'], {key: value for key, value in item.items() if key not in ('name', 'type')})
                for item in schema['inputs']}
    namespace[name].INPUT_TYPES = classmethod(lambda cls, required=required: {'required': required})
namespace['nodes'].NODE_CLASS_MAPPINGS = {name: namespace[name] for name in ('TrimVideoLatent', 'StableCascade_EmptyLatentImage')}

def source_validation(name, arguments):
    inputs = dict(arguments)
    if name == 'TrimVideoLatent':
        inputs['samples'] = {'__value__': {}}
    valid, failures, _ = asyncio.run(namespace['validate_inputs']('oracle', {'node': {'class_type': name, 'inputs': inputs}}, 'node', {}))
    return {'valid': valid, 'errors': sorted(failures, key=lambda item: (item['details'], item['type'])),
            'normalized_inputs': {key: value for key, value in inputs.items() if key != 'samples'}}

for item in schema_gaps:
    item['source_prompt_validation'] = source_validation('TrimVideoLatent', {'trim_amount': cases[item['case_index']]['trim_amount']})
    assert not item['source_prompt_validation']['valid']
    item['label'] = 'source-success/direct negative trim; source prompt and native schema reject'
for item in cascade + cascade_gaps:
    arguments = {'batch_size': 1, **item['arguments']}
    item['source_prompt_validation'] = source_validation('StableCascade_EmptyLatentImage', arguments)


cascade_mapped = []
inputs = {'width': [256, 257], 'height': [256], 'compression': [42, 128, 4], 'batch_size': [1, 2]}
columns = [[], []]
for index in range(max(map(len, inputs.values()))):
    outputs = namespace['StableCascade_EmptyLatentImage'].execute(**namespace['slice_dict'](inputs, index))
    for column, output in zip(columns, outputs):
        column.append(wire(output))
cascade_mapped.append({'node': 'StableCascade_EmptyLatentImage', 'source_repeat_last': True, 'inputs': inputs, 'columns': columns})

result = {'source_sha256': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in FILES},
          'torch': torch.__version__, 'claims': {'SetLatentNoiseMask': 'COMFY-NODE-0618', 'TrimVideoLatent': 'COMFY-NODE-0684', 'StableCascade_EmptyLatentImage': 'COMFY-NODE-0635'},
          'aliases': {'SetLatentNoiseMask': [], 'TrimVideoLatent': [], 'StableCascade_EmptyLatentImage': []}, 'cases': cases, 'representation_gaps': gaps,
          'source_invalid_input_errors': errors, 'mapped_cases': mapped, 'source_schema_gaps': schema_gaps,
          'cascade_cases': cascade, 'cascade_gaps': cascade_gaps, 'cascade_schema': cascade_schema, 'cascade_mapped_cases': cascade_mapped}


def stage_c_source_oracle():
    import contextlib
    import logging
    import os
    import torchvision
    from safetensors.torch import load_file

    checkpoint = Path(os.environ['SIM_STAGE_C_ENCODER_CHECKPOINT'])
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checkpoint_sha256 == 'e23c1fe17ba8cdd6c49a5b292d72502bd4cf729e7de9f2060bb4dcc26737c0a0'
    paths = ['sim/sd.py', 'sim/utils.py', 'sim/ops.py', 'sim/ldm/cascade/stage_c_coder.py']
    management = types.SimpleNamespace(
        intermediate_dtype=lambda: torch.float32,
        throw_exception_if_processing_interrupted=lambda: None,
        cuda_device_context=lambda device: contextlib.nullcontext(),
        load_models_gpu=lambda *arguments, **options: None,
        raise_non_oom=lambda error: (_ for _ in ()).throw(error))
    source_sim = types.SimpleNamespace(model_management=management)
    operations = {'torch': torch, 'sim': source_sim}
    select('sim/ops.py', ['run_every_op', 'CastWeightBiasOp', 'disable_weight_init'], operations)
    model_namespace = {'torch': torch, 'torchvision': torchvision, 'nn': torch.nn,
                       'ops': operations['disable_weight_init']}
    select('sim/ldm/cascade/stage_c_coder.py', ['EfficientNetEncoder', 'StageC_coder'], model_namespace)
    resize_namespace = {'torch': torch}
    select('sim/utils.py', ['common_upscale'], resize_namespace)
    source_sim.utils = types.SimpleNamespace(common_upscale=resize_namespace['common_upscale'])
    node_namespace = {'torch': torch, 'io': io, 'sim': source_sim}
    select('sim_extras/nodes_stable_cascade.py', ['StableCascade_StageC_VAEEncode'], node_namespace)
    io.Image = types.SimpleNamespace(Input=lambda name, **options: {'name': name, 'type': 'IMAGE', **options})
    io.Vae = types.SimpleNamespace(Input=lambda name, **options: {'name': name, 'type': 'VAE', **options})
    schema = node_namespace['StableCascade_StageC_VAEEncode'].define_schema()
    required = {item['name']: (item['type'], {key: value for key, value in item.items() if key not in ('name', 'type')})
                for item in schema['inputs']}
    source_node = node_namespace['StableCascade_StageC_VAEEncode']
    source_node.INPUT_TYPES = classmethod(lambda cls: {'required': required})
    namespace['nodes'].NODE_CLASS_MAPPINGS['StableCascade_StageC_VAEEncode'] = source_node

    tree = ast.parse((ROOT / 'sim/sd.py').read_text())
    original_vae = next(node for node in tree.body if getattr(node, 'name', '') == 'VAE')
    method_names = {'encode', 'throw_exception_if_invalid', 'vae_encode_crop_pixels',
                    'vae_output_dtype', 'spacial_compression_encode'}
    selected_methods = [node for node in original_vae.body if getattr(node, 'name', '') in method_names]
    assert len(selected_methods) == len(method_names)
    original_class = ast.ClassDef(name='SourceVAE', bases=[], keywords=[], body=selected_methods, decorator_list=[])
    ast.fix_missing_locations(original_class)
    vae_namespace = {'torch': torch, 'model_management': management, 'logging': logging,
                     'sim': source_sim}
    exec(compile(ast.Module(body=[original_class], type_ignores=[]), str(ROOT / 'sim/sd.py'), 'exec'), vae_namespace)
    encoder = model_namespace['EfficientNetEncoder']().eval()
    checkpoint_state = load_file(str(checkpoint), device='cpu')
    missing, unexpected = encoder.load_state_dict(checkpoint_state, strict=False)
    assert sorted(missing) == ['mean', 'std'] and not unexpected
    assert len(encoder.backbone) == 8
    assert sum(len(stage) for stage in list(encoder.backbone)[1:-1]) == 40
    stage_class = model_namespace['StageC_coder']
    stage = stage_class.__new__(stage_class)
    torch.nn.Module.__init__(stage)
    stage.encoder = encoder
    stage.eval()
    resource = vae_namespace['SourceVAE']()
    resource.first_stage_model = stage
    resource.downscale_ratio = 32
    resource.crop_input = True
    resource.output_channels = 3
    resource.pad_channel_value = None
    resource.latent_dim = 2
    resource.extra_1d_channel = None
    resource.device = resource.output_device = torch.device('cpu')
    resource.vae_dtype = torch.float32
    resource.disable_offload = False
    resource.process_input = lambda image: image * 2.0 - 1.0
    resource.memory_used_encode = lambda shape, dtype: 1
    resource.patcher = types.SimpleNamespace(get_free_memory=lambda device: 1024 * 1024 * 1024)
    encoder_cases = []
    for shape, compression, offset, label in [
        ((1, 44, 44, 3), 42, 0.0, 'default compression full real encoder'),
        ((1, 47, 61, 4), 42, 0.125, 'non-square center crop and RGB selection'),
        ((2, 64, 64, 3), 64, -0.125, 'two batch items and nondefault compression'),
        ((1, 33, 33, 1), 32, 0.0, 'original grayscale broadcast'),
        ((1, 4, 4, 3), 4, 0.0, 'source-success empty stage-B samples'),
        ((1, 32, 32, 3), 42, 0.0, 'zero resized targets'),
        ((1, 32, 44, 3), 42, 0.0, 'one zero resized target'),
        ((1, 44, 32, 3), 42, 0.0, 'other zero resized target'),
        ((0, 44, 44, 3), 42, 0.0, 'zero batch original VAE result'),
        ((1, 44, 44, 3), 0, 0.0, 'source direct zero compression'),
        ((1, 44, 44, 3), -1, 0.0, 'source direct negative compression'),
    ]:
        image = tensor(shape, offset).add(1.0).mul(0.5)
        item = {'image': wire_tensor(image), 'compression': compression, 'label': label}
        try:
            with torch.inference_mode():
                outputs = node_namespace['StableCascade_StageC_VAEEncode'].execute(image, resource, compression)
            item['outputs'] = [wire(output) for output in outputs]
        except Exception as error:
            item['error_type'] = type(error).__name__
            item['error'] = str(error)
        item['source_prompt_validation'] = source_validation('StableCascade_StageC_VAEEncode',
            {'image': {'__value__': {}}, 'vae': {'__value__': {}}, 'compression': compression})
        encoder_cases.append(item)
    dtype_cases = []
    for dtype, default_dtype, label in [(torch.float64, torch.float32, 'source float64 IMAGE before VAE dtype cast'),
                                        (torch.float32, torch.float64, 'source default-float64 stage-B allocation')]:
        image = tensor((1, 44, 44, 3)).add(1.0).mul(0.5).to(dtype=dtype)
        item = {'image': wire_tensor(image), 'image_dtype': str(dtype),
                'default_dtype': str(default_dtype), 'compression': 42, 'label': label}
        try:
            torch.set_default_dtype(default_dtype)
            with torch.inference_mode():
                outputs = node_namespace['StableCascade_StageC_VAEEncode'].execute(image, resource, 42)
            item['outputs'] = [wire(output) for output in outputs]
            item['output_dtypes'] = [str(output['samples'].dtype) for output in outputs]
        except Exception as error:
            item['error_type'] = type(error).__name__
            item['error'] = str(error)
        finally:
            torch.set_default_dtype(torch.float32)
        dtype_cases.append(item)
    torchvision_source = Path(torchvision.__file__).parent / 'models/efficientnet.py'
    state = encoder.state_dict()
    return {
        'claim': {'StableCascade_StageC_VAEEncode': 'COMFY-NODE-0637'}, 'aliases': [],
        'schema': schema, 'cases': encoder_cases, 'dtype_cases': dtype_cases,
        'source_sha256': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
        'torchvision': torchvision.__version__,
        'torchvision_efficientnet_sha256': hashlib.sha256(torchvision_source.read_bytes()).hexdigest(),
        'state_schema': [{'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype)} for name, value in state.items()],
        'checkpoint': {'url': 'https://huggingface.co/stabilityai/stable-cascade/resolve/e3aee2fd11a00865f5c085d3e741f2e51aef12d3/effnet_encoder.safetensors',
                       'sha256': checkpoint_sha256, 'bytes': checkpoint.stat().st_size,
                       'missing_source_constructor_parameters': sorted(missing), 'unexpected_keys': list(unexpected)},
        'feature_block_count': 40, 'resource_kind': 'original full EfficientNetEncoder eval with actual unmodified StabilityAI checkpoint',
        'execution_policy': 'CPU eval; original ops/node/resize/VAE/StageC encode methods; GPU residency/free-memory callbacks test-controlled; previewer decoding and OOM fallback unexecuted'}



def vae_decode_source_oracle():
    import contextlib
    import logging
    from safetensors.torch import save

    paths = ['nodes.py', 'sim/sd.py', 'sim/pixel_space_convert.py', 'sim/nested_tensor.py']
    models = {'torch': torch}
    select('sim/pixel_space_convert.py', ['PixelspaceConversionVAE'], models)
    select('sim/nested_tensor.py', ['NestedTensor'], models)
    node_namespace = {'torch': torch}
    select('nodes.py', ['VAEDecode'], node_namespace)
    def raise_non_oom(error):
        raise error
    management = types.SimpleNamespace(
        intermediate_dtype=lambda: torch.float32,
        cuda_device_context=lambda device: contextlib.nullcontext(),
        load_models_gpu=lambda *args, **kwargs: None,
        raise_non_oom=raise_non_oom)
    original = next(node for node in ast.parse((ROOT / 'sim/sd.py').read_text()).body
                    if getattr(node, 'name', '') == 'VAE')
    methods = {'decode', 'throw_exception_if_invalid', 'vae_output_dtype'}
    selected = [node for node in original.body if getattr(node, 'name', '') in methods]
    assert len(selected) == len(methods)
    wrapper = ast.ClassDef(name='SourceDecodeVAE', bases=[], keywords=[], body=selected, decorator_list=[])
    ast.fix_missing_locations(wrapper)
    wrappers = {'torch': torch, 'model_management': management, 'logging': logging}
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), str(ROOT / 'sim/sd.py'), 'exec'), wrappers)
    model = models['PixelspaceConversionVAE']().eval()
    resource = wrappers['SourceDecodeVAE']()
    resource.first_stage_model = model
    resource.latent_dim = 2
    resource.device = resource.output_device = torch.device('cpu')
    resource.vae_dtype = torch.float32
    resource.disable_offload = True
    resource.memory_used_decode = lambda shape, dtype: 1
    resource.patcher = types.SimpleNamespace(get_free_memory=lambda device: 1024 * 1024 * 1024)
    constructor = next(node for node in original.body if getattr(node, 'name', '') == '__init__')
    output_policy = next(node for node in constructor.body if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Attribute) and target.attr == 'process_output' for target in node.targets))
    exec(compile(ast.Module(body=[output_policy], type_ignores=[]), str(ROOT / 'sim/sd.py'), 'exec'), {'self': resource})
    entries = []
    def record_decode(samples, label, kind='tensor'):
        entry = {'label': label, 'kind': kind}
        if isinstance(samples, dict) and 'samples' in samples:
            value = samples['samples']
            entry['input'] = ([wire_tensor(part) for part in value.unbind()] if getattr(value, 'is_nested', False)
                              else wire_tensor(value))
            entry['metadata'] = {key: value for key, value in samples.items() if key != 'samples'}
        try:
            images = node_namespace['VAEDecode']().decode(resource, samples)[0]
            entry['output'] = wire_tensor(images)
        except Exception as error:
            entry['error_type'] = type(error).__name__
            entry['error'] = str(error)
        entries.append(entry)
    for shape, offset, label in [
        ((1, 3, 2, 4), 0.0, 'non-square original pixel-space VAE'),
        ((2, 3, 3, 5), 0.25, 'two batch images'),
        ((1, 3, 2, 4), 3.0, 'upper clamp'),
        ((1, 3, 2, 4), -3.0, 'lower clamp'),
        ((1, 3, 2, 2, 4), 0.0, 'source 2D VAE takes first video frame; canonical admission gap'),
        ((1, 1, 2, 4), 0.0, 'source grayscale pixel latent; canonical channel gap'),
        ((1, 4, 2, 4), 0.0, 'source four-channel pixel latent; canonical channel gap'),
        ((0, 3, 2, 4), 0.0, 'source empty batch failure'),
        ((1, 3, 0, 2, 4), 0.0, 'source empty video frame failure'),
    ]:
        record_decode({'samples': tensor(shape, offset), 'batch_index': [99], 'source_marker': 'retained ignored metadata'}, label)
    record_decode({'samples': models['NestedTensor']([tensor((1, 3, 2, 2, 4)), tensor((1, 4, 6))])},
                  'source ordered nested video/audio chooses video first; canonical 2D frame admission gap', 'nested')
    record_decode({'samples': models['NestedTensor']([tensor((1, 3, 2, 4)), tensor((1, 3, 3, 5))])},
                  'source nested first rank-four image; canonical nested rank gap', 'nested')
    record_decode({'samples': models['NestedTensor']([])}, 'source empty nested failure', 'nested')
    record_decode({}, 'source missing samples key failure', 'malformed')
    record_decode([], 'source wrong dictionary failure', 'malformed')
    artifact = save(model.state_dict())
    return {'claim': {'VAEDecode': 'COMFY-NODE-0710'},
            'schema': node_namespace['VAEDecode'].INPUT_TYPES(),
            'aliases': node_namespace['VAEDecode'].SEARCH_ALIASES,
            'cases': entries,
            'source_sha256': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
            'artifact': {'producer': 'original PixelspaceConversionVAE constructor state_dict via safetensors0.4.2',
                         'bytes': list(artifact), 'sha256': hashlib.sha256(artifact).hexdigest()},
            'execution_policy': 'original node, nested owner, complete source pixel-space model and original VAE.decode wrapper; CPU F32/output F32; residency/free-memory callbacks controlled; no OOM retry or full video decoder resource exercised',
            'unproved': ['actual full learned image/video decoder resources', 'rank-five decoded image flattening with a real video VAE', 'non-F32 model/intermediate policies', 'actual UI/OS-process restart']}

result['stage_c_encoder'] = stage_c_source_oracle()
result['vae_decode'] = vae_decode_source_oracle()

content = json.dumps(result, indent=2, allow_nan=False) + '\n'
parser = argparse.ArgumentParser()
parser.add_argument('--check', action='store_true')
arguments = parser.parse_args()
path = Path(__file__).with_name('source-contract.json')
if arguments.check:
    assert path.read_text() == content, 'Independent original-Torch fixture drift'
else:
    path.write_text(content)
print('actual Torch source', len(cases), 'cases', sum('output' in case for case in cases), 'success',
      sum('error' in case for case in cases), 'errors', len(gaps), 'representation gaps', len(errors), 'invalid-input failures', len(mapped), 'mapped lists')

print('StableCascade actual source', len(cascade), 'cases', sum('outputs' in case for case in cascade), 'success', sum('error' in case for case in cascade), 'errors;', len(cascade_gaps), 'source-success schema/dtype gaps;', len(schema_gaps), 'retained original trim schema gaps')
