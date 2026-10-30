import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--repository", type=Path)
arguments = parser.parse_args()
source = (arguments.repository or Path(__file__).resolve().parents[5]) / "projects/sim/ComfyUI/sim/k_diffusion/sampling.py"
expected = json.loads(Path(__file__).with_name("adaptive-zero-reference.json").read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected["source_sha256"]
class Tensor(np.ndarray):
    def log(self):return np.log(self).view(Tensor)
    def neg(self):return (-self).view(Tensor)
    def exp(self):return np.exp(self).view(Tensor)
    def expm1(self):return np.expm1(self).view(Tensor)
    def abs(self):return np.abs(self).view(Tensor)
    def new_ones(self,shape):return np.ones(shape,dtype=self.dtype).view(Tensor)
    def numel(self):return self.size

def tensor(value):return np.asarray(value,dtype=np.float32).view(Tensor)
namespace={'math':math,'nn':SimpleNamespace(Module=object),'torch':SimpleNamespace(
    tensor=tensor,minimum=lambda a,b:np.minimum(a,b).view(Tensor),
    maximum=lambda a,b:np.maximum(a,b).view(Tensor),
    linalg=SimpleNamespace(norm=lambda value:np.linalg.norm(value).view(Tensor)))}
nodes=[node for node in ast.parse(source.read_text()).body if isinstance(node,ast.ClassDef) and node.name in ['PIDStepSizeController','DPMSolver']]
assert len(nodes)==2
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
class ProbeBound(Exception):pass
cases=[]
for parameters in [{},{'rtol':0.0},{'atol':0.0},{'accept_safety':0.0},{'h_init':0.0},{'rtol':0.0,'atol':0.0}]:
    trace=[]
    def record(info):
        trace.append({key:int(info[key]) for key in ['steps','nfe','n_accept','n_reject']})
        if len(trace)>=64:raise ProbeBound()
    solver=namespace['DPMSolver'](lambda x,sigma:x*np.float32(.2)+np.float32(.1),info_callback=record)
    with np.errstate(all='ignore'):
        try:
            output,info=solver.dpm_solver_adaptive(tensor([1.,-.5]),solver.t(tensor(2.)),solver.t(tensor(.5)),noise_sampler=lambda a,b:tensor([0.,0.]),**parameters)
        except ProbeBound:
            case={'parameters':parameters,'status':'source_did_not_converge_within_64_probe_callbacks','info':trace[-1]}
        else:
            assert np.isfinite(output).all()
            case={'parameters':parameters,'status':'complete','output':output.tolist(),'info':info}
    cases.append(case)
result={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'provenance':'Pinned DPMSolver/PID AST, NumPy 2.3.5 float32 adapter; temporary 64-callback probe bound; not PyTorch certification','cases':cases}
assert result == expected, "Pinned adaptive source boundary reference changed"
print("PASS: four finite adaptive source trajectories and two bounded non-convergence cases")
