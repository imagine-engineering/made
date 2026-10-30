from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[5] / "projects/sim/ComfyUI"

def node_reference():
    import ast
    import copy
    import hashlib
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np


    class Tensor(np.ndarray):
        @property
        def device(self): return 'cpu'
        def clone(self): return self.copy().view(Tensor)
        def new_zeros(self, shape): return np.zeros(shape, dtype=self.dtype).view(Tensor)
        def abs(self): return np.abs(self).view(Tensor)

    def tensor(values, dtype=np.float32): return np.asarray(values, dtype=dtype).view(Tensor)
    torch = SimpleNamespace(
        Tensor=Tensor,
        arange=lambda start,end: tensor(np.arange(start,end),np.int64),
        flip=lambda value,axes: np.flip(value,axes).view(Tensor),
        cat=lambda values: np.concatenate(values).view(Tensor),
        linspace=lambda start,end,steps,device=None: np.linspace(start,end,steps,dtype=np.float32).view(Tensor),
        floor=lambda value: np.floor(value).view(Tensor),
    )
    namespace = {'torch':torch, 'io':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *outputs:outputs),
        'sim':SimpleNamespace(samplers=SimpleNamespace(ksampler=lambda name,parameters:{'algorithm':name,'parameters':parameters}))}
    def select(path, classes=(), functions=()):
        source=ROOT/path
        nodes=[node for node in ast.parse(source.read_text()).body if
            isinstance(node,ast.ClassDef) and node.name in classes or
            isinstance(node,ast.FunctionDef) and node.name in functions]
        assert len(nodes)==len(classes)+len(functions)
        future=ast.parse('from __future__ import annotations').body
        exec(compile(ast.Module(body=future+nodes,type_ignores=[]),str(source),'exec'),namespace)

    select('sim/k_diffusion/sa_solver.py',functions=['get_tau_interval_func'])
    namespace['sa_solver']=SimpleNamespace(get_tau_interval_func=namespace['get_tau_interval_func'])
    select('sim/model_sampling.py',functions=['time_snr_shift'])
    flow_class=next(node for node in ast.parse((ROOT/'sim/model_sampling.py').read_text()).body if isinstance(node,ast.ClassDef) and node.name=='ModelSamplingDiscreteFlow')
    flow_methods=[node for node in flow_class.body if isinstance(node,ast.FunctionDef) and node.name in ['sigma','percent_to_sigma']]
    flow=ast.ClassDef(name='Flow',bases=[],keywords=[],body=flow_methods,decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[flow],type_ignores=[])),'<pinned flow methods>','exec'),namespace)
    class Model:
        def __init__(self,shift=3.0):
            self.sampling=namespace['Flow']()
            self.sampling.shift=shift
            self.sampling.multiplier=1000.0
            self.sampling.sigma_min=tensor(shift*.001/(1+(shift-1)*.001))
            self.sampling.sigma_max=tensor(1.)
            self.guidance=None
        def get_model_object(self,name):
            assert name=='model_sampling'
            return self.sampling
        def clone(self): return copy.copy(self)
        def set_model_sampler_cfg_function(self,function): self.guidance=function

    names=['SamplerLMS','SamplerSASolver','SamplerSEEDS2','SamplingPercentToSigma','SDTurboScheduler','SetFirstSigma','SplitSigmas','SplitSigmasDenoise']
    select('sim_extras/nodes_custom_sampler.py',classes=names)
    select('sim_extras/nodes_video_model.py',classes=['VideoLinearCFGGuidance','VideoTriangleCFGGuidance'])
    cases=[]
    for values in [[],[0.],[14.,4.,1.,0.],[10.,5.,3.,2.,1.,0.]]:
        for step in [0,1,3,10000]:
            result=namespace['SplitSigmas'].execute(tensor(values),step)
            cases.append({'class_type':'SplitSigmas','inputs':{'sigmas':values,'step':step},'outputs':[item.tolist() for item in result]})
        for denoise in [0.,.125,.5,.625,1.]:
            result=namespace['SplitSigmasDenoise'].execute(tensor(values),denoise)
            cases.append({'class_type':'SplitSigmasDenoise','inputs':{'sigmas':values,'denoise':denoise},'outputs':[item.tolist() for item in result]})
        for sigma in [0.,136.,20000.]:
            inputs={'sigmas':values,'sigma':sigma}
            try: result=namespace['SetFirstSigma'].execute(tensor(values),sigma)
            except IndexError: cases.append({'class_type':'SetFirstSigma','inputs':inputs,'error':'empty sigma sequence'})
            else: cases.append({'class_type':'SetFirstSigma','inputs':inputs,'outputs':[item.tolist() for item in result]})
    for shift in [1.,3.]:
        model=Model(shift)
        for steps in [1,4,10]:
            for denoise in [0.,.05,.1,.25,.99,1.]:
                result=namespace['SDTurboScheduler'].execute(model,steps,denoise)
                cases.append({'class_type':'SDTurboScheduler','inputs':{'model_shift':shift,'steps':steps,'denoise':denoise},'outputs':[item.tolist() for item in result]})
        for percent in [0.,.001,.2,.5,.8,1.]:
            for actual in [False,True]:
                result=namespace['SamplingPercentToSigma'].execute(model,percent,actual)
                cases.append({'class_type':'SamplingPercentToSigma','inputs':{'model_shift':shift,'sampling_percent':percent,'return_actual_sigma':actual},'outputs':[float(item) for item in result]})
    for order in [1,4,100]:
        cases.append({'class_type':'SamplerLMS','inputs':{'order':order},'outputs':namespace['SamplerLMS'].execute(order)})
    for solver_type in ['phi_1','phi_2']:
        for eta,noise,ratio in [(0.,0.,.01),(1.,1.,.5),(100.,100.,1.)]:
            inputs={'solver_type':solver_type,'eta':eta,'s_noise':noise,'r':ratio}
            cases.append({'class_type':'SamplerSEEDS2','inputs':inputs,'outputs':namespace['SamplerSEEDS2'].execute(**inputs)})
    for start,end in [(0.,1.),(.2,.8),(.8,.2)]:
        inputs={'eta':1.5,'sde_start_percent':start,'sde_end_percent':end,'s_noise':2.,'predictor_order':3,'corrector_order':4,'use_pece':True,'simple_order_2':False}
        model=Model(3.)
        output=namespace['SamplerSASolver'].execute(model,**inputs)[0]
        tau=output['parameters'].pop('tau_func')
        samples=[0.,.001,.1,.5,.9,1.,10.]
        output['tau_samples']=[[sigma,tau(sigma)] for sigma in samples]
        output['start_sigma']=model.sampling.percent_to_sigma(start)
        output['end_sigma']=model.sampling.percent_to_sigma(end)
        cases.append({'class_type':'SamplerSASolver','inputs':{'model_shift':3.,**inputs},'outputs':[output]})
    for name in ['VideoLinearCFGGuidance','VideoTriangleCFGGuidance']:
        for batch in [1,2,5]:
            for minimum in [0.,1.,9.]:
                model=Model()
                changed=namespace[name]().patch(model,minimum)[0]
                assert model.guidance is None
                cond=tensor(np.arange(batch).reshape(batch,1,1,1)+2)
                uncond=tensor(np.arange(batch).reshape(batch,1,1,1)*.25)
                result=changed.guidance({'cond':cond,'uncond':uncond,'cond_scale':7.})
                cases.append({'class_type':name,'inputs':{'min_cfg':minimum,'batch':batch,'conditional':cond.flatten().tolist(),'unconditional':uncond.flatten().tolist(),'scale':7.},'outputs':result.flatten().tolist()})
    paths=['sim_extras/nodes_custom_sampler.py','sim_extras/nodes_video_model.py','sim/model_sampling.py','sim/k_diffusion/sa_solver.py']
    result={'reference_runtime':'NumPy 2.3.5 adapter; pinned node AST and Flow sigma/percent methods, not PyTorch runtime certification','source_sha256':{path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths},'cases':cases}
    return result

def seeds_reference():
    import ast
    import hashlib
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from functools import partial
    import numpy as np
    source=ROOT/'sim/k_diffusion/sampling.py'
    class Tensor(np.ndarray):
        def __getitem__(self,key):
            value=super().__getitem__(key)
            return np.asarray(value,dtype=self.dtype).view(Tensor)
        def new_ones(self,shape):return np.ones(shape,dtype=self.dtype).view(Tensor)
        def log(self):return np.log(self).view(Tensor)
        def neg(self):return np.negative(self).view(Tensor)
        def exp(self):return np.exp(self).view(Tensor)
        def expm1(self):return np.expm1(self).view(Tensor)
        def clone(self):return self.copy().view(Tensor)

    def tensor(values):return np.asarray(values,dtype=np.float32).view(Tensor)
    torch=SimpleNamespace(Tensor=Tensor,no_grad=lambda:lambda function:function,lerp=lambda start,end,weight:start+(end-start)*weight,expm1=lambda value:np.expm1(value).view(Tensor))
    namespace={'torch':torch,'partial':partial,'trange':lambda count,disable=None:range(count),'sim':SimpleNamespace(model_sampling=SimpleNamespace(CONST=type('Constant',(),{})))}
    names={'sample_seeds_2','sigma_to_half_log_snr','half_log_snr_to_sigma','offset_first_sigma_for_snr','ei_h_phi_1','ei_h_phi_2'}
    nodes=[node for node in ast.parse(source.read_text()).body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    class Model:
        def __init__(self):
            self.inner_model=SimpleNamespace(model_patcher=SimpleNamespace(get_model_object=lambda name:SimpleNamespace(noise_scale=1.0)))
            self.calls=0
        def __call__(self,latent,sigma,**kwargs):
            self.calls+=1
            return tensor([.3 if self.calls==1 else .25])
    cases=[]
    for solver in ['phi_1','phi_2']:
        model=Model()
        result=namespace['sample_seeds_2'](model,tensor([1.]),tensor([2.,1.]),eta=0.,s_noise=0.,noise_sampler=lambda first,second:(_ for _ in ()).throw(AssertionError('deterministic case requested noise')),r=.5,solver_type=solver)
        assert model.calls==2
        cases.append({'class_type':'SamplerSEEDS2','inputs':{'solver_type':solver,'eta':0.,'s_noise':0.,'r':.5},'initial':[1.],'sigmas':[2.,1.],'denoised':[[.3],[.25]],'expected':result.tolist()})
    result={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'runtime':'pinned AST, NumPy float32 adapter, deterministic VE profile','cases':cases}
    return result

def discrete_reference():
    import ast
    import hashlib
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np

    class Tensor(np.ndarray):
        @property
        def device(self):return 'cpu'
        def __getitem__(self,key):return np.asarray(super().__getitem__(key)).view(Tensor)
        def float(self):return np.asarray(self,dtype=np.float32).view(Tensor)
        def long(self):return np.asarray(self,dtype=np.int64).view(Tensor)
        def to(self,device):assert device=='cpu';return self
        def log(self):return np.log(self).view(Tensor)
        def exp(self):return np.exp(self).view(Tensor)
        def floor(self):return np.floor(self).view(Tensor)
        def ceil(self):return np.ceil(self).view(Tensor)
        def frac(self):return (self-np.trunc(self)).view(Tensor)
    class Module:
        def register_buffer(self,name,value):setattr(self,name,value)
    torch=SimpleNamespace(nn=SimpleNamespace(Module=Module),float64=np.float64,float32=np.float32,
        linspace=lambda start,end,steps,dtype:np.linspace(start,end,steps,dtype=dtype).view(Tensor),
        tensor=lambda value:np.asarray(value,dtype=np.float32).view(Tensor),
        zeros=lambda shape,dtype:np.zeros(shape,dtype=dtype).view(Tensor),
        cumprod=lambda value,dim:np.cumprod(value,axis=dim).view(Tensor),
        clamp=lambda value,min,max:np.clip(value,min,max).view(Tensor))
    namespace={'torch':torch,'np':np}
    def load(path,names):
        source=ROOT/path
        nodes=[node for node in ast.parse(source.read_text()).body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
        assert len(nodes)==len(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    load('sim/ldm/modules/diffusionmodules/util.py',['make_beta_schedule'])
    load('sim/model_sampling.py',['ModelSamplingDiscrete'])
    namespace['sim']=SimpleNamespace(model_sampling=SimpleNamespace(ModelSamplingDiscrete=namespace['ModelSamplingDiscrete']))
    load('sim_extras/nodes_model_advanced.py',['ModelSamplingDiscreteDistilled'])
    cases=[]
    for name in ['ModelSamplingDiscrete','ModelSamplingDiscreteDistilled']:
        model=namespace[name]()
        for percent in [0.,.001,.2,.5,.8,.99,.999,1.]:
            for actual in [False,True]:
                sigma=model.percent_to_sigma(percent)
                if actual and percent==0:sigma=model.sigma_max.item()
                if actual and percent==1:sigma=model.sigma_min.item()
                cases.append({'profile':'lcm' if name.endswith('Distilled') else 'sd15','percent':percent,'return_actual_sigma':actual,'sigma':float(sigma)})
    paths=['sim/ldm/modules/diffusionmodules/util.py','sim/model_sampling.py','sim_extras/nodes_model_advanced.py']
    result={'source_sha256':{path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths},'runtime':'pinned discrete and distilled source classes with NumPy float32/float64 tensor adapter','cases':cases}
    return result

fixture = json.loads(Path(__file__).with_name("fixture.json").read_text())
for key, reference in [("source_reference", node_reference), ("sampler_consumption_reference", seeds_reference), ("discrete_percent_reference", discrete_reference)]:
    assert fixture[key] == json.loads(json.dumps(reference())), key
print("Verified 138 node cases, two SEEDS2 trajectories, and 32 discrete percentage values")
