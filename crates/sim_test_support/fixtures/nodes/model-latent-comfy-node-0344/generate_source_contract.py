"""Independent original Torch node oracle; native LATENT representational limits stay explicit."""
import argparse,ast,copy,hashlib,json,logging,math,sys
from pathlib import Path
from types import ModuleType,SimpleNamespace
import torch
import numpy as np
from enum import Enum
from typing import Optional
ROOT=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI');DEST=Path(__file__).with_name('source-contract.json');torch.set_num_threads(1)
FILES=['nodes.py','sim_extras/nodes_lt.py','sim_extras/nodes_latent.py','sim_extras/nodes_rebatch.py','sim/nested_tensor.py'];NODES={'RepeatLatentBatch':('nodes.py','RepeatLatentBatch','repeat'),'LTXVConcatAVLatent':('sim_extras/nodes_lt.py','LTXVConcatAVLatent','execute'),'LTXVSeparateAVLatent':('sim_extras/nodes_lt.py','LTXVSeparateAVLatent','execute'),'RebatchLatents':('sim_extras/nodes_rebatch.py','LatentRebatch','execute'),'ReplaceVideoLatentFrames':('sim_extras/nodes_latent.py','ReplaceVideoLatentFrames','execute')}
namespace={'__builtins__':__builtins__,'torch':torch,'math':math,'logging':logging,'nodes':SimpleNamespace(MAX_RESOLUTION=16384),'io':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *values:values)}
nested=ModuleType('pinned_nested');exec(compile((ROOT/FILES[4]).read_text(),str(ROOT/FILES[4]),'exec'),nested.__dict__);namespace['sim']=SimpleNamespace(nested_tensor=nested)
for file,class_name,method in NODES.values():
 original=next(node for node in ast.parse((ROOT/file).read_text()).body if isinstance(node,ast.ClassDef) and node.name==class_name);node=copy.deepcopy(original);node.body=[member for member in node.body if not isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)) or member.name not in ['define_schema','INPUT_TYPES']]
 for member in ast.walk(node):
  if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
   member.returns=None
   for argument in [*member.args.args,*member.args.kwonlyargs]:argument.annotation=None
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(ROOT/file),'exec'),namespace)
def tensor(shape,start=0.,constant=None):
 count=math.prod(shape);values=[float(constant) if constant is not None else float(start+index/16) for index in range(count)];return {'shape':shape,'values':values}
def latent(shape=[2,2,2,2],start=0.,mask=None,indices=None,metadata=None):
 output={'samples':tensor(shape,start)}
 if mask is not None:output['noise_mask']=mask
 if indices is not None:output['batch_index']=indices
 output.update(metadata or {});return output
def decode(value):
 if isinstance(value,dict) and set(value)=={'shape','values'}:return torch.tensor(value['values'],dtype=torch.float32).reshape(value['shape'])
 if isinstance(value,dict) and set(value)=={'nested'}:return nested.NestedTensor([decode(item) for item in value['nested']])
 if isinstance(value,dict):return {key:decode(item) for key,item in value.items()}
 if isinstance(value,list):return [decode(item) for item in value]
 return value
def encode(value):
 if isinstance(value,torch.Tensor):return {'shape':list(value.shape),'values':value.contiguous().flatten().tolist()}
 if isinstance(value,nested.NestedTensor):return {'nested':[encode(item) for item in value.tensors]}
 if isinstance(value,dict):return {key:encode(item) for key,item in value.items()}
 if isinstance(value,(tuple,list)):return [encode(item) for item in value]
 return value
cases=[];errors=[]
class Recorder(logging.Handler):
 def __init__(self):super().__init__();self.events=[]
 def emit(self,record):self.events.append({'level':record.levelname,'message':record.getMessage()})
def observe(node,arguments,scope='supported canonical LATENT fields'):
 values=decode(arguments);original=encode(values);recorder=Recorder();logging.getLogger().addHandler(recorder)
 try:
  _,class_name,method=NODES[node];target=namespace[class_name]() if node=='RepeatLatentBatch' else namespace[class_name];result=getattr(target,method)(**values)
 except Exception as error:errors.append({'node':node,'input':arguments,'scope':scope,'error_type':type(error).__name__,'error_message':str(error),'input_after':encode(values),'logs':recorder.events})
 else:
  identity=[next((name for name,item in values.items() if output is item),None) for output in result];cases.append({'node':node,'input':arguments,'scope':scope,'output':encode(result),'identity':identity,'input_after':encode(values),'logs':recorder.events})
 finally:logging.getLogger().removeHandler(recorder)
 if encode(values)!=original:raise RuntimeError('source mutated input dictionary/tensor')
for shape in [[2,2,2,2],[2,2,3,2,2]]:
 for amount in [1,2,3]:
  for mask in [None,tensor([1,1,3,3],constant=.5),tensor([2,1,3,3],constant=.25),tensor([2,1,3,3],constant=1),tensor([3,1,3,3],constant=.75)]:observe('RepeatLatentBatch',{'samples':latent(shape,mask=mask,indices=[3,7],metadata={'downscale_ratio_spacial':8}),'amount':amount})
observe('RepeatLatentBatch',{'samples':latent([3,2,2,2],mask=tensor([2,1,3,3],constant=.5),indices=[2,4,6]),'amount':2})
observe('RepeatLatentBatch',{'samples':latent(indices=[]),'amount':2},'source empty index error; canonical source index passthrough')
for masks in [(None,None),(tensor([1,1,2,2,2],constant=.25),None),(None,tensor([1,1,5],constant=.75)),(tensor([1,1,2,2,2],constant=.25),tensor([1,1,5],constant=.75))]:
 video=latent([1,2,2,2,2],start=1,mask=masks[0],indices=[3],metadata={'downscale_ratio_spacial':8,'downscale_ratio_temporal':4,'sample_rate':24000});audio=latent([1,2,5],start=4,mask=masks[1],indices=[9],metadata={'type':'audio','sample_rate':48000});observe('LTXVConcatAVLatent',{'video_latent':video,'audio_latent':audio})
 av={'samples':{'nested':[video['samples'],audio['samples']]},'batch_index':[9],'type':'audio','sample_rate':48000,'downscale_ratio_spacial':8,'downscale_ratio_temporal':4}
 if masks[0] is not None or masks[1] is not None:av['noise_mask']={'nested':[masks[0] or tensor([1,2,2,2,2],constant=1),masks[1] or tensor([1,2,5],constant=1)]}
 observe('LTXVSeparateAVLatent',{'av_latent':av})
observe('LTXVSeparateAVLatent',{'av_latent':latent([2,2,2,2],metadata={'sample_rate':24000})})
observe('LTXVSeparateAVLatent',{'av_latent':latent([1,2,2,2])})
for index in [-8,-5,-4,-3,-1,0,2,3,4,5]:
 for source_frames in [1,2,4]:observe('ReplaceVideoLatentFrames',{'destination':latent([2,2,4,2,2],metadata={'downscale_ratio_spacial':8}),'source':latent([2,2,source_frames,2,2],start=8,mask=tensor([1,1,2,2],constant=.5),metadata={'downscale_ratio_spacial':16}),'index':index})
observe('ReplaceVideoLatentFrames',{'destination':latent([1,2,4,2,2]),'source':None,'index':100})
observe('ReplaceVideoLatentFrames',{'destination':latent([2,2,4,2,2]),'source':latent([1,1,2,1,1],start=8),'index':1})
observe('ReplaceVideoLatentFrames',{'destination':latent([2,2,4,2,2]),'source':latent([2,3,2,2,2]),'index':1})
for batch_size in [1,2,3,4,5]:
 for masks in ['none','fraction','one']:
  items=[latent([2,2,2,2],mask=None if masks=='none' else tensor([1,1,2,2],constant=.5 if masks=='fraction' else 1),metadata={'downscale_ratio_spacial':16}),latent([3,2,2,2],start=4,mask=None if masks=='none' else tensor([1,1,2,2],constant=.5 if masks=='fraction' else 1),indices=[20,21,22])]
  observe('RebatchLatents',{'latents':items,'batch_size':[batch_size,7]})
observe('RebatchLatents',{'latents':[latent([3,2,2,2]),latent([2,2,3,2],start=4)],'batch_size':[3]})
observe('RebatchLatents',{'latents':[latent([2,2,2,2],mask={'shape':[1,1,2,2],'values':[0.,2.,0.,2.]})],'batch_size':[2]})
observe('RebatchLatents',{'latents':[],'batch_size':[2]})
observe('RebatchLatents',{'latents':[latent([2,2,2,2])],'batch_size':[]})
observe('RebatchLatents',{'latents':[latent([2,2,2,2],indices=[4,8])],'batch_size':[0]})
observe('RebatchLatents',{'latents':[latent([2,2,2,2],mask=tensor([1,1,2,2],constant=.5)),latent([2,2,2,2])],'batch_size':[2]})
observe('RepeatLatentBatch',{'samples':latent(indices=[7]),'amount':2},'source nonnegative index length passthrough')
observe('LTXVConcatAVLatent',{'video_latent':latent([1,2,2,2,2],indices=[3],metadata={'sample_rate':24000,'downscale_ratio_spacial':8}),'audio_latent':latent([1,2,5],start=4)},'source absent audio key preserves video metadata')
observe('LTXVSeparateAVLatent',{'av_latent':{'samples':{'nested':[tensor([1,2,2,2,2]),tensor([1,2,5])]},'batch_index':[9,12]}},'source nonnegative index length passthrough')
observe('ReplaceVideoLatentFrames',{'destination':latent([2,2,4,2,2]),'source':latent([2,2,1,2,2],start=8,indices=[99]),'index':1},'source nonnegative index length passthrough')
observe('RebatchLatents',{'latents':[latent(indices=[3])],'batch_size':[2]},'source nonnegative index length passthrough')
observe('RebatchLatents',{'latents':[latent(indices=[])] ,'batch_size':[2]},'source empty index list is retained')
observe('RebatchLatents',{'latents':[latent(indices=[3]),latent(start=4)],'batch_size':[2]},'source processed offset follows index count')
supported_count=len(cases)
observe('LTXVConcatAVLatent',{'video_latent':{**latent([1,2,2,2,2],metadata={'sample_rate':24000}),'noise_mask':None},'audio_latent':latent([1,2,5],metadata={'sample_rate':None})},'unsupported canonical key-presence capability: present-null noise_mask and metadata override absent')
observe('LTXVSeparateAVLatent',{'av_latent':{'samples':{'nested':[tensor([1,2,2,2,2]),tensor([1,2,5])]},'noise_mask':None}},'unsupported canonical key-presence capability: shallow copies retain present-null mask')
capability_gaps=cases[supported_count:];del cases[supported_count:]
result={'sources':{file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in FILES},'torch_version':torch.__version__,'cases':cases,'errors':errors,'capability_gaps':capability_gaps,'scope':'Actual original node AST and NestedTensor with real TorchCPU tensors; dictionaries, masks, batch indices, metadata, logs, identity and unchanged inputs. Native model/audio/load dependencies and representational limits remain separate.'}
# Original node and first-stage getters are executed without reconstructing their equations.
empty_file='sim_extras/nodes_lt_audio.py';geometry_file='sim/ldm/lightricks/vae/audio_vae.py'
original=next(node for node in ast.parse((ROOT/empty_file).read_text()).body if isinstance(node,ast.ClassDef) and node.name=='LTXVEmptyLatentAudio');node=copy.deepcopy(original);node.body=[member for member in node.body if not isinstance(member,ast.FunctionDef) or member.name!='define_schema']
for member in ast.walk(node):
 if isinstance(member,ast.FunctionDef):
  member.returns=None
  for argument in [*member.args.args,*member.args.kwonlyargs]:argument.annotation=None
namespace['sim'].model_management=SimpleNamespace(intermediate_device=lambda:torch.device('cpu'))
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(ROOT/empty_file),'exec'),namespace)
original=next(node for node in ast.parse((ROOT/geometry_file).read_text()).body if isinstance(node,ast.ClassDef) and node.name=='AudioVAE');geometry=copy.deepcopy(original);geometry.name='OriginalAudioGeometry';geometry.bases=[ast.Name(id='object',ctx=ast.Load())];geometry.body=[member for member in geometry.body if isinstance(member,ast.FunctionDef) and member.name in ['num_of_latents_from_frames','sample_rate','mel_hop_length','mel_bins','latent_channels','latent_frequency_bins','latents_per_second']]
namespace['LATENT_DOWNSAMPLE_FACTOR']=4
exec(compile(ast.fix_missing_locations(ast.Module(body=[geometry],type_ignores=[])),str(ROOT/geometry_file),'exec'),namespace)
def empty_observe(frames,rate,batch,channels=8,mel_bins=64,hop=160,dtype=torch.float32):
 stage=namespace['OriginalAudioGeometry']();stage.autoencoder=SimpleNamespace(sampling_rate=16000,mel_hop_length=hop,mel_bins=mel_bins,decoder=SimpleNamespace(z_channels=channels));wrapper=SimpleNamespace(first_stage_model=stage,latent_channels=stage.latent_channels)
 arguments={'frames_number':frames,'frame_rate':rate,'batch_size':batch};configuration={'sample_rate':16000,'mel_hop_length':hop,'mel_bins':mel_bins,'latent_channels':channels,'wrapper_sample_rate':44100,'default_dtype':str(dtype)};previous=torch.get_default_dtype();torch.set_default_dtype(dtype)
 try:
  output=namespace['LTXVEmptyLatentAudio'].execute(**arguments,audio_vae=wrapper);return {'input':arguments,'configuration':configuration,'output':encode(output),'dtype':str(output[0]['samples'].dtype)}
 except Exception as error:return {'input':arguments,'configuration':configuration,'error_type':type(error).__name__,'error_message':str(error)}
 finally:torch.set_default_dtype(previous)
result['empty_audio_sources']={file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in [empty_file,geometry_file]}
result['empty_audio_cases']=[empty_observe(frames,rate,batch) for frames,rate,batch in [(1,25,1),(1,26,2),(2,3,1),(25,24,1),(97,25,1),(999,1000,1),(3,1000,3),(1000,1000,1)]]
result['empty_audio_errors']=[empty_observe(1,0,1),empty_observe(-1,25,1),empty_observe(1,25,-1)]
result['empty_audio_capability_cases']=[empty_observe(0,25,1),empty_observe(1,25,0),empty_observe(1,25,1,channels=4,mel_bins=80,hop=200),empty_observe(1,25,1,dtype=torch.float64)]
model_files=['sim/ldm/lightricks/vae/pixel_norm.py','sim/ldm/lightricks/vae/causal_audio_autoencoder.py','sim/ldm/lightricks/vocoders/vocoder.py']
model_namespace={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'Optional':Optional,'Enum':Enum,'logging':logging,'math':math,'np':np,'sim':SimpleNamespace(ops=SimpleNamespace(disable_weight_init=torch.nn),model_management=SimpleNamespace())}
for file in model_files:
 tree=ast.parse((ROOT/file).read_text());tree.body=[member for member in tree.body if not isinstance(member,(ast.Import,ast.ImportFrom))];exec(compile(tree,str(ROOT/file),'exec'),model_namespace)
with torch.device('meta'):
 autoencoder=model_namespace['CausalAudioAutoencoder']();vocoder=model_namespace['Vocoder']()
state=[{'name':prefix+name,'shape':list(value.shape),'dtype':str(value.dtype)} for prefix,module in [('autoencoder.',autoencoder),('vocoder.',vocoder)] for name,value in module.state_dict().items()]
result['empty_audio_model']={'sources':{file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in model_files},'configuration':{'audio_vae':autoencoder.get_default_config(),'vocoder':vocoder.get_default_config()},'state':state,'state_count':len(state),'contiguous_f32_bytes':4*sum(math.prod(item['shape']) for item in state),'provenance':'Unchanged original source constructor/class/function AST on Torch meta device. Import scaffolding supplies standard Torch nn parameter factories in place of disabled-weight-init subclasses; only state names/shapes/dtypes are observed, never native pixels. Canonical recipe tests create actual zero-valued state tensors; empty-node geometry is independent of weights.'}
def original_empty_audio_prompt_validation():
    import ast
    import asyncio
    import collections
    import copy
    import dataclasses
    import enum
    import hashlib
    import inspect
    import io as standard_io
    import json
    from pathlib import Path
    import shutil
    import sys
    import traceback
    import types
    import typing
    import typing_extensions
    import torch

    root = Path('/home/asvegah/repos/sim/projects/sim/ComfyUI')
    module = types.ModuleType('task631_original_io')
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update(vars(typing))
    namespace.update(vars(dataclasses))
    namespace['__name__'] = 'task631_original_io'
    namespace.update(ABC=__import__('abc').ABC, abstractmethod=__import__('abc').abstractmethod,
                     copy=copy, inspect=inspect, Counter=collections.Counter, Iterable=collections.abc.Iterable,
                     Enum=enum.Enum, torch=torch, final=typing_extensions.final,
                     NotRequired=typing_extensions.NotRequired, asyncio=asyncio, BytesIO=standard_io.BytesIO,
                     Path=Path, shutil=shutil)
    files = []

    def source_ast(relative, names=None):
        path = root / relative
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
                     and (names is None or getattr(node, 'name', None) in names)]
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in ('Curve', 'Range'):
                node.body = [member for member in node.body if not isinstance(member, (ast.Import, ast.ImportFrom))]
        tree.body.insert(0, ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0))
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
        files.append(relative)

    source_ast('sim_api/internal/__init__.py')
    source_ast('sim_api/latest/_util/geometry_types.py')
    source_ast('sim_api/latest/_util/image_types.py')
    namespace['_SVG'] = namespace['SVG']
    source_ast('sim_api/latest/_io.py')
    namespace['io'] = module
    source_ast('sim_extras/nodes_lt_audio.py', {'LTXVEmptyLatentAudio'})
    source_ast('sim_execution/graph.py', {'get_input_info'})
    source_ast('execution.py', {'validate_inputs', 'full_type_name'})
    node = namespace['LTXVEmptyLatentAudio']
    namespace['_io'] = module
    namespace['nodes'] = types.SimpleNamespace(NODE_CLASS_MAPPINGS={'LTXVEmptyLatentAudio': node})
    namespace['sys'] = sys
    namespace['traceback'] = traceback
    cases = []
    for frames, rate, batch in [(1,25,1),(0,25,1),(1,25,0),(1,0,1),(-1,25,1),(1,25,-1)]:
        inputs = {'frames_number': frames, 'frame_rate': rate, 'batch_size': batch, 'audio_vae': None}
        prompt = {'audio': {'class_type': 'LTXVEmptyLatentAudio', 'inputs': inputs}}
        result = asyncio.run(namespace['validate_inputs']('source-prompt-proof', prompt, 'audio', {}))
        cases.append({'input': {name:value for name,value in inputs.items() if name != 'audio_vae'},
                      'valid': result[0], 'errors': result[1]})
    return {'sources': {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest() for relative in files}, 'schema': node.INPUT_TYPES(), 'cases': cases, 'scope': 'Unchanged original V3 Int/Schema/ComfyNode binding and original validate_inputs/get_input_info AST. Only import scaffolding is removed; original schema construction, finalization and bounds checks execute. The VAE literal is null and never executed; this probe observes prompt scalar validation only, not retained-resource execution.'}

result['empty_audio_prompt_validation']=original_empty_audio_prompt_validation()
def original_load_latent_contract():
    import os
    import tempfile
    import safetensors.torch
    source = ROOT / 'nodes.py'
    original = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef) and node.name == 'LoadLatent')
    with tempfile.TemporaryDirectory(prefix='task631-load-latent-oracle-') as temporary:
        directory = Path(temporary)
        namespace = {'os': os, 'hashlib': hashlib, 'safetensors': safetensors,
                     'folder_paths': SimpleNamespace(get_input_directory=lambda: str(directory),
                         get_annotated_filepath=lambda name: str(directory / name),
                         exists_annotated_filepath=lambda name: (directory / name).is_file())}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[])), str(source), 'exec'), namespace)
        node = namespace['LoadLatent']
        successes = []
        errors = []
        capabilities = []
        def observe(name, tensors=None, encoded=None, capability=False):
            path = directory / name
            if encoded is not None:
                path.write_bytes(encoded)
            elif tensors is not None:
                safetensors.torch.save_file(tensors, path)
            encoded = path.read_bytes() if path.is_file() else None
            case = {'node': 'LoadLatent', 'input': {'latent': name},
                    'encoded_bytes': list(encoded) if encoded is not None else None,
                    'encoded_sha256': hashlib.sha256(encoded).hexdigest() if encoded is not None else None,
                    'validate_inputs': node.VALIDATE_INPUTS(name)}
            if encoded is not None:
                case['change_token'] = node.IS_CHANGED(name)
                if case['change_token'] != case['encoded_sha256']:
                    raise RuntimeError('Original file cache identity mismatch')
            try:
                output = node().load(name)
                case['output'] = encode(output)
                case['dtype'] = str(output[0]['samples'].dtype)
                (capabilities if capability else successes).append(case)
            except Exception as error:
                case['error_type'] = type(error).__name__
                case['error_message'] = str(error).replace(str(directory), '<source-temp>')
                errors.append(case)
            return case
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int64,
                      torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32]:
            for marker in [False, True]:
                values = [0., .25, 1., 7.] if dtype == torch.uint8 else [-.5, 0., .25, 1.]
                samples = torch.tensor(values, dtype=dtype).reshape(1,1,2,2)
                tensors = {'latent_tensor': samples}
                if marker:
                    tensors['latent_format_version_0'] = torch.tensor([])
                observe(f'{str(dtype)}-{marker}.latent', tensors)
        for name, shape, marker in [('audio', [2,3,4], False), ('video', [1,2,2,3,4], True),
                                    ('nonempty-marker', [1,1,2,2], True), ('not-latent', [1,1,2,2], False)]:
            samples = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) / 7.0 - .75
            tensors = {'latent_tensor': samples, 'ignored_tensor': torch.ones([2,3])}
            if marker:
                tensors['latent_format_version_0'] = torch.tensor([99], dtype=torch.int64)
            observe(name + ('.txt' if name == 'not-latent' else '.latent'), tensors)
        observe('empty-samples.latent', {'latent_tensor': torch.empty([0,1,2,2])}, capability=True)
        observe('rank-two.latent', {'latent_tensor': torch.ones([2,2])}, capability=True)
        for dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            observe(str(dtype)+'.latent', {'latent_tensor': torch.tensor([-.5,0.,.25,1.]).to(dtype).reshape(1,1,2,2)}, capability=True)
        float64_case = next(case for case in successes if case['input']['latent'] == 'torch.float64-False.latent')
        encoded_complex = bytes(float64_case['encoded_bytes']).replace(b'"F64"', b'"C64"')
        observe('unsupported-complex.latent', encoded=encoded_complex)
        observe('missing-tensor.latent', {'other_tensor': torch.zeros([1,1,2,2])})
        observe('bad-header.latent', encoded=b'not-a-safetensors')
        original_bytes = bytes(successes[0]['encoded_bytes'])
        observe('truncated.latent', encoded=original_bytes[:-1])
        observe('empty-file.latent', encoded=b'')
        # Source validates the provided annotated name, before body file I/O.
        missing_validation = node.VALIDATE_INPUTS('missing-file.latent')
        (directory / 'directory.latent').mkdir()
        options = node.INPUT_TYPES()
        import asyncio
        import contextvars
        validation_namespace = dict(vars(sys.modules['task631_original_io']))
        validation_namespace.update(namespace)
        validation_namespace['nodes'] = SimpleNamespace(NODE_CLASS_MAPPINGS={'LoadLatent': node})
        validation_namespace['contextvars'] = contextvars
        prompt_sources = []
        for relative, names in [
            ('sim_execution/graph_utils.py', {'is_link', 'ExecutionBlocker'}),
            ('sim_execution/utils.py', None),
            ('sim_execution/graph.py', {'get_input_info'}),
            ('execution.py', {'get_input_data', 'resolve_map_node_over_list_results', '_async_map_node_over_list', 'validate_inputs', 'full_type_name'}),
        ]:
            tree = ast.parse((ROOT / relative).read_text())
            tree.body = [member for member in tree.body if not isinstance(member, (ast.Import, ast.ImportFrom))
                         and (names is None or getattr(member, 'name', None) in names)]
            tree.body.insert(0, ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0))
            exec(compile(ast.fix_missing_locations(tree), str(ROOT / relative), 'exec'), validation_namespace)
            prompt_sources.append(relative)
        prompt_cases = []
        for name in [successes[0]['input']['latent'], 'not-latent.txt', 'missing-file.latent']:
            prompt = {'latent': {'class_type': 'LoadLatent', 'inputs': {'latent': name}}}
            result = asyncio.run(validation_namespace['validate_inputs']('source-load-latent-proof', prompt, 'latent', {}))
            prompt_cases.append({'input': {'latent': name}, 'valid': result[0], 'errors': result[1]})
        prompt_validation = {'sources': {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() for relative in prompt_sources},
                             'cases': prompt_cases, 'scope': 'Original prompt validation/input-list mapper/current-node context AST with unchanged LoadLatent VALIDATE_INPUTS. Import scaffolding only; no custom-validation or combo-choice algorithm is replaced. The actual input directory is the transparent source temporary boundary.'}
        return {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'safetensors_version': safetensors.__version__, 'safetensors_torch_sha256': hashlib.sha256(Path(safetensors.torch.__file__).read_bytes()).hexdigest(), 'cases': successes, 'errors': errors,
                'capability_gaps': capabilities, 'source_ui_input_types': options,
                'missing_file_validation': missing_validation, 'prompt_validation': prompt_validation, 'search_aliases': node.SEARCH_ALIASES,
                'scope': 'Unchanged complete original LoadLatent AST with actual safetensors file save/load and Torch CPU. Annotated filename resolution is a transparent trusted temporary input-directory boundary; actual INPUT_TYPES/IS_CHANGED/VALIDATE_INPUTS execute. Only temporary error-path address text is normalized, never bytes, source values, schema or arithmetic. Empty/rank-two samples remain canonical LATENT representation gaps, separate from supported native cases.'}

result['load_latent_contract'] = original_load_latent_contract()
parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();encoded=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
if args.check:
 if DEST.read_text()!=encoded:raise SystemExit('source oracle drift')
else:DEST.write_text(encoded)
print(f"Latent partfive actual source: {len(cases)} successes,{len(errors)} errors")
