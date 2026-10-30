from pathlib import Path
import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import struct
import sys
import types

FORMAT = 'sim.source.conditioning-part-three.original-cpu-f32.v1'
SOURCE = Path('/home/asvegah/repos/sim/projects/sim/ComfyUI')
PINS = {
    'nodes.py': 'b8dfdde1de8975be762b085048143cc2dda8fc9202695e460ecc2c8dfe44bc4b',
    'node_helpers.py': '22317a033cce556331bfedc5f4ab6c2833184fd2bbdf9496c9182a4973a7098a',
}
# Remaining file pins are inserted by the static preparation, never by execution.
ADDITIONAL_PINS = {'sim_extras/nodes_video_model.py': 'b3d355bc0878fd5d3068ee54edb94fd08ac0f657dafe448654d530a289838947', 'sim_extras/nodes_audio.py': 'f9da5427304479919a743475750e89ec7b72f460aa4ab5fccbed08b788d910f4', 'sim/samplers.py': 'd882256ae9baa1d23f1367ab2ec3b021fdc15fe39ce4cb49ea2c1ee10026a649', 'sim/conds.py': '72058e9a22c972a9c875819e59d432d30d367fd2f7092ee6c6c45e5a60c959b0'}
ADDITIONAL_PINS['sim/model_sampling.py'] = '8559ac2f700b788babf30cc16f410c93890e1f71485ac8f6a299ea36e2cb8717'
ADDITIONAL_PINS['sim_api/latest/_io.py'] = 'ac986092cace7600461b8547e21f770126958d8c15eef59d79d6709cca4b7717'
ADDITIONAL_PINS['sim/ldm/modules/diffusionmodules/util.py'] = 'fb58652a35521fc23bdcb75d91adace8e4cc79e2d5b13af1617a38d0c0f7142e'
ROWS = [
    ('0092', 'nodes', 'ConditioningSetArea', 'append'),
    ('0093', 'nodes', 'ConditioningSetAreaPercentage', 'append'),
    ('0094', 'sim_extras.nodes_video_model', 'ConditioningSetAreaPercentageVideo', 'append'),
    ('0095', 'nodes', 'ConditioningSetAreaStrength', 'append'),
    ('0100', 'nodes', 'ConditioningSetTimestepRange', 'set_range'),
    ('0101', 'sim_extras.nodes_audio', 'ConditioningStableAudio', 'execute'),
    ('0103', 'nodes', 'ConditioningZeroOut', 'zero_out'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def definitions():
    cases = []
    def add(name, row, arguments, pool='absent', gap=None, dims=(16, 16), nonfinite=False):
        cases.append(dict(name=name, row=row, arguments=arguments, pool=pool,
                          gap=gap, dimensions=list(dims), nonfinite=nonfinite))
    for name, strength, width, height, x, y in [
        ('area-f64-point-one', 0.1, 65, 79, 17, 9),
        ('area-negative-zero', -0.0, 64, 64, 0, 0),
        ('area-zero', 0.0, 64, 64, 0, 0),
        ('area-adjacent-f64', math.nextafter(0.1, math.inf), 64, 64, 0, 0),
        ('area-max-offset', 0.1, 64, 64, 16384, 0),
    ]:
        add(name, '0092', dict(width=width, height=height, x=x, y=y, strength=strength))
    for name, width, height, x, y, gap in [
        ('percentage-point-one', 0.5, 0.75, 0.25, 0.0, None),
        ('percentage-zero-endpoint', 0.0, 0.0, 1.0, 1.0, 'native-zero-extent-or-endpoint-consumer'),
        ('percentage-halfway-round', 0.15625, 0.21875, 0.03125, 0.09375, None),
        ('percentage-full-negative-zero', 1.0, 1.0, -0.0, -0.0, None),
    ]:
        add(name, '0093', dict(width=width, height=height, x=x, y=y, strength=0.1), gap=gap)
    for name, temporal, height, width, z, y, x, gap in [
        ('video-point-one', 0.5, 0.5, 0.75, 0.25, 0.0, 0.125, None),
        ('video-zero-endpoint', 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 'native-zero-extent-or-endpoint-consumer'),
        ('video-halfway-round', 0.3125, 0.15625, 0.21875, 0.0625, 0.03125, 0.09375, None),
    ]:
        add(name, '0094', dict(temporal=temporal, height=height, width=width, z=z, y=y, x=x, strength=0.1), dims=(8, 16, 16), gap=gap)
    expanded = struct.unpack('<f', struct.pack('<f', 0.1))[0]
    for name, strength in [
        ('strength-point-one', 0.1), ('strength-f32-expanded', expanded),
        ('strength-adjacent-f64', math.nextafter(0.1, math.inf)),
        ('strength-negative-zero', -0.0), ('strength-zero', 0.0),
        ('strength-ten', 10.0), ('strength-f32-underflow', 1e-46),
        ('strength-tiny', 2.0 ** -54),
    ]:
        add(name, '0095', dict(strength=strength))
    for name, start, end, gap in [
        ('range-zero', 0.0, 0.0, 'native-equal-window-consumer'),
        ('range-one', 1.0, 1.0, 'native-equal-window-consumer'),
        ('range-reversed', 0.8, 0.2, 'native-reversed-window-consumer'),
        ('range-full', 0.0, 1.0, None),
    ]:
        add(name, '0100', dict(start=start, end=end), gap=gap)
    for name, start, total in [
        ('audio-zero', 0.0, 0.0), ('audio-reversed', 5.0, 0.0),
        ('audio-default', 0.0, 47.0), ('audio-point-one', 0.1, 0.1),
    ]:
        add(name, '0101', dict(seconds_start=start, seconds_total=total))
    for pool in ['absent', 'none', 'some']:
        add('zero-' + pool, '0103', {}, pool=pool)
    add('zero-nonfinite', '0103', {}, pool='some', nonfinite=True)
    add('zero-lyrics', '0103', {}, pool='some', gap='native-lyrics-metadata-unrepresented')
    add('zero-empty-list', '0103', {}, gap='native-empty-conditioning-unrepresented')
    return cases


def producer():
    if platform.python_version() != '3.12.14':
        raise RuntimeError('Prepared Source authority requires Python 3.12.14')
    pins = {**PINS, **ADDITIONAL_PINS}
    for name, expected in pins.items():
        if sha(SOURCE / name) != expected:
            raise RuntimeError('Pinned Source changed: ' + name)
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0, str(SOURCE))
    sys.argv = ['task618-original', '--cpu', '--fp32-text-enc', '--disable-dynamic-vram', '--disable-xformers']
    import sim.options
    sim.options.enable_args_parsing()
    import torch
    import node_helpers
    from sim_api.latest import IO
    import sim.samplers as samplers
    import sim.conds as condition_owners
    import sim.model_sampling as sampling_owner
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if str(torch.__version__) != '2.8.0+cpu':
        raise RuntimeError('This prepared profile requires genuine Torch 2.8.0+cpu')
    modules = {name: importlib.import_module(name) for _, name, _, _ in ROWS}
    if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:
        raise RuntimeError('Original import changed the required one-thread profile')
    for module in [node_helpers, samplers, condition_owners, sampling_owner, *modules.values()]:
        actual = Path(module.__file__).resolve()
        if not actual.is_relative_to(SOURCE) or actual.relative_to(SOURCE).as_posix() not in pins:
            raise RuntimeError('Unexpected original module owner: ' + str(actual))
    if sys.byteorder != 'little':
        raise RuntimeError('Prepared exact word capture requires little-endian CPU')
    def tensor_record(value):
        if value.dtype != torch.float32 or str(value.device) != 'cpu':
            raise RuntimeError('Non-CPU/F32 capture attempted')
        raw = value.detach().contiguous().numpy().tobytes()
        return dict(shape=list(value.shape), dtype='F32', device='CPU',
                    words32=[word[0] for word in struct.iter_unpack('<I', raw)],
                    raw_le_sha256=hashlib.sha256(raw).hexdigest())
    def value_record(value):
        if isinstance(value, torch.Tensor):
            return {'tensor': tensor_record(value)}
        if isinstance(value, float):
            return {'f64_bits': struct.unpack('<Q', struct.pack('<d', value))[0]}
        if isinstance(value, tuple):
            return {'tuple': [value_record(item) for item in value]}
        if isinstance(value, list):
            return {'list': [value_record(item) for item in value]}
        if value is None or isinstance(value, (str, bool, int)):
            return value
        raise TypeError('Unsupported captured metadata type: ' + type(value).__name__)
    def pack(entries):
        return [dict(primary=tensor_record(primary), metadata={name:value_record(value) for name,value in metadata.items()})
                for primary,metadata in entries]
    def make_entries(case, negative=False):
        values = [-3.5, 0.1, -0.0, 0.0, 1.7, 0.25]
        if negative:
            values = [0.25, -0.0, 1.7, -3.5, 0.0, 0.1]
        if case['nonfinite']:
            values = [math.inf, -math.inf, math.nan, -0.0, 0.0, 0.1]
        primary = torch.tensor(values, dtype=torch.float32).reshape(1,3,2)
        metadata = {'strength':0.5}
        if case['pool'] == 'none':
            metadata['pooled_output'] = None
        if case['pool'] == 'some':
            metadata['pooled_output'] = torch.tensor([[-3.5,0.1]], dtype=torch.float32)
        if case['name'] == 'zero-lyrics':
            metadata['lyrics'] = torch.tensor([[0.1,-3.5]], dtype=torch.float32)
        return [] if case['name'] == 'zero-empty-list' else [[primary, metadata]]
    sampling = sampling_owner.ModelSamplingDiscrete()
    def source_sampler(entries, dimensions):
        if any('seconds_start' in metadata for _,metadata in entries):
            return {'status':'unrun-audio-model-consumer-required', 'scope':'node timing metadata only; no fabricated StableAudio conditioning injection'}
        if not entries:
            return {'status':'unrun-empty-list', 'scope':'no entry to consume'}
        conditions = [dict(metadata, model_conds={'c_crossattn':condition_owners.CONDCrossAttn(primary)}, uuid='task618-source-entry')
                      for primary,metadata in entries]
        x = torch.zeros((1,1,*dimensions), dtype=torch.float32)
        before = pack(entries)
        try:
            samplers.resolve_areas_and_cond_masks_multidim(conditions, dimensions, torch.device('cpu'))
            temporal = any('start_percent' in condition for condition in conditions)
            observations = []
            if temporal:
                # This object only carries the genuine sampler attribute for the unchanged helper.
                # It is not an original learned ModelBase or ModelPatcher.
                samplers.calculate_start_end_timesteps(types.SimpleNamespace(model_sampling=sampling), conditions)
                metadata = conditions[0]
                percentages = [0.0, 0.5, 1.0, metadata['start_percent'], metadata['end_percent']]
                timestep_values = [sampling.percent_to_sigma(percent) for percent in percentages]
            else:
                percentages = [None]
                timestep_values = [0.5]
            for percent, timestep_value in zip(percentages, timestep_values):
                timestep = torch.tensor([timestep_value], dtype=torch.float32)
                results = []
                for condition in conditions:
                    result = samplers.get_area_and_mult(condition, x, timestep)
                    results.append(None if result is None else dict(
                        area=result.area, input_x=tensor_record(result.input_x), mult=tensor_record(result.mult),
                        conditioning={name:tensor_record(value.cond) for name,value in result.conditioning.items()}))
                observations.append(dict(percent=value_record(percent), source_sigma=value_record(timestep_value),
                                         timestep=tensor_record(timestep), results=results))
            return {'status':'success', 'results':observations[0]['results'], 'observations':observations,
                    'resolved_timing':[{name:value_record(value) for name,value in condition.items() if name in ['timestep_start','timestep_end']} for condition in conditions],
                    'timing_owner':'genuine default ModelSamplingDiscrete; attribute-only SimpleNamespace model carrier' if temporal else None,
                    'scope':'genuine Source area/timing/get_area_and_mult and real CONDCrossAttn; no learned model/diffusion/audio-model execution'}
        except Exception as error:
            return {'status':'error', 'type':type(error).__name__, 'message':str(error), 'scope':'original sampler boundary'}
        finally:
            if pack(entries) != before:
                raise RuntimeError('Source sampler unexpectedly mutated node output')
    identities = []
    for feature, module_name, class_name, method_name in ROWS:
        cls = getattr(modules[module_name], class_name)
        if feature == '0101':
            if cls.define_schema().node_id != class_name:
                raise RuntimeError('Source V3 schema identity changed')
        elif modules[module_name].NODE_CLASS_MAPPINGS.get(class_name) is not cls:
            raise RuntimeError('Original registered class identity changed')
        method = getattr(cls, method_name)
        identities.append(dict(feature='COMFY-NODE-'+feature, class_name=class_name, module=module_name,
            method=method_name, class_line=inspect.getsourcelines(cls)[1],
            class_sha256=hashlib.sha256(inspect.getsource(cls).encode()).hexdigest(),
            method_sha256=hashlib.sha256(inspect.getsource(method).encode()).hexdigest()))
    rows = {feature:(module, cls, method) for feature,module,cls,method in ROWS}
    captured = []
    for case in definitions():
        module_name,class_name,method_name = rows[case['row']]
        cls = getattr(modules[module_name], class_name)
        positive = make_entries(case)
        negative = make_entries(case, True) if case['row']=='0101' else None
        inputs = [positive] if negative is None else [positive,negative]
        before = [pack(column) for column in inputs]
        arguments = dict(case['arguments'])
        arguments.update(conditioning=positive) if negative is None else arguments.update(positive=positive,negative=negative)
        returned = getattr(cls if case['row']=='0101' else cls(),method_name)(**arguments)
        columns = returned.args if isinstance(returned, IO.NodeOutput) else returned
        if [pack(column) for column in inputs] != before:
            raise RuntimeError('Original node mutated input: ' + case['name'])
        aliases = [[dict(primary_input=next((index for index,column in enumerate(inputs) if any(item[0] is primary for item in column)),None),
                         copied_metadata=all(metadata is not item[1] for column in inputs for item in column))
                    for primary,metadata in column] for column in columns]
        captured.append(dict(case, class_type=class_name, feature='COMFY-NODE-'+case['row'],
            input=before, output=[pack(column) for column in columns], aliases=aliases,
            returned_type=type(returned).__module__+'.'+type(returned).__qualname__,
            argument_bits={name:value_record(value) for name,value in case['arguments'].items()},
            sampler=source_sampler(columns[0],case['dimensions'])))
    for name,expected in pins.items():
        if sha(SOURCE/name)!=expected:
            raise RuntimeError('Source changed during capture: '+name)
    versions = {name:importlib.metadata.version(name) for name in ['torch','numpy','Pillow','safetensors','av','torchaudio','torchvision','transformers','torchsde']}
    return dict(format=FORMAT, producer_sha256=sha(__file__), source_sha256=pins,
        python_version=platform.python_version(), package_versions=versions,
        source_imports={name:dict(path=str(Path(module.__file__).resolve().relative_to(SOURCE)), sha256=sha(module.__file__))
                        for name,module in sorted(sys.modules.items()) if getattr(module,'__file__',None) and Path(module.__file__).resolve().is_relative_to(SOURCE)},
        profile='genuine Torch CPU/F32 little-endian, one Torch/BLAS thread', machine=platform.machine(), identities=identities,
        discrete_sampling_sigmas=tensor_record(sampling.sigmas),
        cases=captured, case_count=len(captured),
        qualification='Genuine imported Source node classes/helpers/IO/Torch and original sampler area/mult methods; synthetic tiny resources; no learned model, full prompt execution, mask/ControlNet, or native parity inferred.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    if output.suffix != '.json':
        raise ValueError('Explicit JSON capture path required')
    if not args.check and output.exists():
        raise FileExistsError('Refusing to replace existing capture: '+str(output))
    if args.check and not output.is_file():
        raise FileNotFoundError('Required capture absent: '+str(output))
    value=producer()
    encoded=(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n').encode()
    if args.check:
        if output.read_bytes()!=encoded:
            raise RuntimeError('Original numerical repeat differs; existing capture retained')
    else:
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('xb') as stream:
            stream.write(encoded)
    print(json.dumps(dict(path=str(output),sha256=hashlib.sha256(encoded).hexdigest(),cases=value['case_count'],check=args.check)))


if __name__ == '__main__':
    main()
