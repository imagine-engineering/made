import ast, hashlib, json, math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

torch.set_num_threads(1)
repository=Path(__file__).resolve().parents[5]
root=repository/'projects/sim/ComfyUI'; fingerprints={}
def source(relative,names,extra=None):
 path=root/relative;fingerprints[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
 namespace={'torch':torch,'np':np,'math':math,'io':SimpleNamespace(ComfyNode=object,NodeOutput=lambda value:value)}
 if extra:namespace.update(extra)
 selected=[]
 for node in ast.parse(path.read_text()).body:
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:selected.append(node)
  elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets):selected.append(node)
 exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body+selected,type_ignores=[]),str(path),'exec'),namespace)
 return namespace
kd=source('sim/k_diffusion/sampling.py',{'append_zero','get_sigmas_karras','get_sigmas_exponential'})
custom=source('sim_extras/nodes_custom_sampler.py',{'KarrasScheduler','ExponentialScheduler','FlipSigmas','ExtendIntermediateSigmas'}, {'k_diffusion_sampling':SimpleNamespace(**kd)})
flux=source('sim_extras/nodes_flux.py',{'generalized_time_snr_shift','compute_empirical_mu','get_schedule','Flux2Scheduler'})
ideo=source('sim_extras/nodes_ideogram4.py',{'_LOGSNR_MIN','_LOGSNR_MAX','_logit_normal_schedule','ideogram4_sigmas','Ideogram4Scheduler'})
gits=source('sim_extras/nodes_gits.py',{'NOISE_LEVELS','loglinear_interp','GITSScheduler'})
cases=[]
def capture(name,owner,parameters):
 converted={key:torch.tensor(value,dtype=torch.float32) if key=='sigmas' else value for key,value in parameters.items()}
 try:
  result=owner[name].execute(**converted)
  cases.append({'class_type':name,'inputs':parameters,'f32_bits':result.view(torch.int32).tolist(),'finite':bool(torch.isfinite(result).all())})
 except (ValueError,ZeroDivisionError,KeyError,IndexError,RuntimeError) as error:
  cases.append({'class_type':name,'inputs':parameters,'source_error':type(error).__name__})
for steps in [1,2,7,20,41]:
 for minimum,maximum in [(0.0291675,14.614642),(0.,10.),(10.,0.),(1.,1.),(20.,1.)]:
  for rho in [0.,0.01,0.5,1.,7.,100.]:capture('KarrasScheduler',custom,dict(steps=steps,sigma_min=minimum,sigma_max=maximum,rho=rho))
  capture('ExponentialScheduler',custom,dict(steps=steps,sigma_min=minimum,sigma_max=maximum))
for sigmas in [[],[0.],[3.],[10.,7.,2.,0.],[0.,2.,1.]]:
 capture('FlipSigmas',custom,dict(sigmas=sigmas))
 for steps in [1,2,7]:
  for start,end in [(-1.,0.),(7.,2.),(1.,12.)]:
   for spacing in ['linear','cosine','sine']:capture('ExtendIntermediateSigmas',custom,dict(sigmas=sigmas,steps=steps,start_at_sigma=start,end_at_sigma=end,spacing=spacing))
for steps in [1,2,7,20,200]:
 for width,height in [(16,16),(1024,1024),(1048,1050),(2048,2048),(16384,16384)]:capture('Flux2Scheduler',flux,dict(steps=steps,width=width,height=height))
for steps in [1,2,7,20,200]:
 for width,height in [(256,256),(1024,1024),(8192,8192)]:
  for mu,std in [(-10.,0.1),(0.,1.75),(10.,5.)]:capture('Ideogram4Scheduler',ideo,dict(steps=steps,width=width,height=height,mu=mu,std=std))
for coeff in sorted(gits['NOISE_LEVELS']):
 for steps in [2,3,20,21,41]:
  for denoise in [0.,0.1,0.5,1.]:capture('GITSScheduler',gits,dict(coeff=coeff,steps=steps,denoise=denoise))
result={'provenance':'Actual pinned source AST numerical execution with PyTorch CPU; no source tensor-math shim','torch_version':torch.__version__,'numpy_version':np.__version__,'sources':fingerprints,'cases':cases,'gits_noise_levels':gits['NOISE_LEVELS']}
expected=json.loads(Path(__file__).with_name('source_schedules.json').read_text())
assert json.loads(json.dumps(result, allow_nan=False)) == expected, 'Pinned source schedules or reference environment changed; fixtures were not modified'
print('Verified',len(cases),'actual-source cases;',sum('source_error' in case for case in cases),'source errors;',sum(not case.get('finite',True) for case in cases),'nonfinite results; PyTorch',torch.__version__)
