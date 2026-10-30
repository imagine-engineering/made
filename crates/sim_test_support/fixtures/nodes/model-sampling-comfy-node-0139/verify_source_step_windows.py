import ast,json,hashlib
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[5]
source=root/'projects/sim/ComfyUI/sim/samplers.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='KSampler')
namespace={'torch':torch,'SCHEDULER_NAMES':['normal'],'SAMPLER_NAMES':['euler'],'sampler_object':lambda name:name}
def sample(model,noise,positive,negative,cfg,device,sampler,sigmas,model_options,**kwargs):
 return {'sigmas':sigmas.tolist()}
namespace['sample']=sample
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
klass=namespace['KSampler'];cases=[]
for sigmas in [[],[0.],[10.,7.,2.,0.]]:
 for start in [None,0,1,3,10000]:
  for end in [None,0,1,3,10000]:
   for force in [False,True]:
    instance=klass.__new__(klass);instance.sigmas=torch.tensor(sigmas,dtype=torch.float32);instance.sampler='euler';instance.model=None;instance.device='cpu';instance.model_options={}
    out=instance.sample(torch.zeros(1),None,None,1.,latent_image='latent',start_step=start,last_step=end,force_full_denoise=force)
    cases.append({'sigmas':sigmas,'start_step':start,'last_step':end,'force_full_denoise':force,'output':out})
result={'provenance':'Pinned KSampler.sample control flow with real PyTorch tensors; model sampling replaced only by a recorder to distinguish dispatch from unchanged-latent branches. This does not claim model inference.', 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':cases}
path=root/'crates/sim_test_support/fixtures/nodes/model-sampling-sim-node-0139/source_step_windows.json'
assert json.loads(json.dumps(result)) == json.loads(path.read_text()), 'pinned step-window source changed'
print(len(cases),'source step-window cases verified')
