#!/usr/bin/env python3
"""Execute pinned sampling prediction and TrainSampler loss/VJP methods."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[5]
SOURCES = {
    'projects/sim/ComfyUI/sim/model_sampling.py': '8559ac2f700b788babf30cc16f410c93890e1f71485ac8f6a299ea36e2cb8717',
    'projects/sim/ComfyUI/sim_extras/nodes_train.py': 'd88aabbc72da32e1e5d934b6e4f3a6587438fe7a7840199771530644a21916ae',
}


def selected_source(path, names, class_method=None):
    raw = (ROOT/path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCES[path]:
        raise ValueError(f'pinned source changed: {path}')
    tree = ast.parse(raw)
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if class_method:
        selected += [method for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_method[0]
                     for method in node.body if isinstance(method, ast.FunctionDef) and method.name == class_method[1]]
    if len(selected) != len(names) + bool(class_method):
        raise ValueError(f"source symbols incomplete: {path}")
    return selected


def generate():
    namespace = {'torch': torch}
    nodes = selected_source(next(iter(SOURCES)), {'reshape_sigma','EPS','V_PREDICTION','V_PREDICTION_DDPM','EDM','CONST','X0','IMG_TO_IMG','IMG_TO_IMG_FLOW','COSMOS_RFLOW'})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned model_sampling.py','exec'),namespace)
    methods = selected_source('projects/sim/ComfyUI/sim_extras/nodes_train.py', {'make_batch_extra_option_dict'}, ('TrainSampler','fwd_bwd'))
    exec(compile(ast.Module(body=methods,type_ignores=[]),'pinned nodes_train.py','exec'),namespace)
    rows=[]
    for prediction, source_class in [('epsilon','EPS'),('v_prediction','V_PREDICTION'),('v_prediction_ddpm','V_PREDICTION_DDPM'),('edm','EDM'),('constant','CONST'),('denoised','X0'),('image_to_image','IMG_TO_IMG'),('image_to_image_flow','IMG_TO_IMG_FLOW'),('cosmos_r_flow','COSMOS_RFLOW')]:
        policy=namespace[source_class]()
        policy.sigma_data=0.5 if prediction=='edm' else 1.0
        policy.noise_scale=1.0
        for sigma in (0.05,0.5,2.0):
            noise=torch.tensor([0.125,-0.75,1.25,-0.375])
            latent=torch.tensor([0.3,-0.2,0.75,-0.625])
            raw=torch.tensor([0.2,-0.4,0.6,-0.8],requires_grad=True)
            sigmas=torch.tensor([sigma])
            xt=policy.noise_scaling(sigmas,noise.clone(),latent.clone(),False)
            x0=policy.noise_scaling(torch.zeros_like(sigmas),torch.zeros_like(noise),latent.clone(),False)
            class Wrapper:
                inner_model=SimpleNamespace(model_sampling=policy)
                conds={}
                def __call__(self, values, times, **kwargs):
                    self.scaled_input=policy.calculate_input(times,values)
                    self.denoised=policy.calculate_denoised(times,raw,values)
                    return self.denoised
            wrapper=Wrapper()
            sampler=SimpleNamespace(loss_fn=torch.nn.MSELoss(),training_dtype=torch.float32,grad_acc=2,grad_scaler=None)
            loss=namespace['fwd_bwd'](sampler,wrapper,sigmas,noise.clone(),latent.clone(),[{'fixture':'conditioning'}],[0],{},1)
            rows.append({'prediction':prediction,'sigma':sigma,'sigma_data':policy.sigma_data,'noise':noise.tolist(),'latent':latent.tolist(),'raw':raw.detach().tolist(),'noisy':xt.tolist(),'target':x0.tolist(),'scaled_input':wrapper.scaled_input.detach().tolist(),'denoised':wrapper.denoised.detach().tolist(),'loss':loss.item(),'gradient':raw.grad.tolist()})
    return {'schema_version':1,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sources':SOURCES,'symbols':{node.name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest() for node in nodes+methods},'environment':{'torch':torch.__version__,'threads':torch.get_num_threads()},'scope':'Actual pinned prediction classes and unchanged TrainSampler.fwd_bwd; explicit raw-output parameter and noise/latent inputs, MSE and grad_acc2. No full model, source RNG, optimizer or BF16 equivalence claim.','absolute_tolerance':2e-6,'relative_tolerance':2e-5,'cases':rows}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    encoded=(json.dumps(generate(),indent=2,sort_keys=True)+'\n').encode()
    output=Path(__file__).with_name('training-policy-source-oracle.json')
    if args.check:
        if output.read_bytes()!=encoded: raise ValueError('source training policy oracle differs')
    else: output.write_bytes(encoded)


if __name__=='__main__': main()
