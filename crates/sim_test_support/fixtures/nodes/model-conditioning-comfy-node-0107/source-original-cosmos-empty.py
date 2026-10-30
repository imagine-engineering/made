from pathlib import Path
import hashlib,json,os,sys
root=Path('/home/asvegah/repos/sim');source=root/'projects/sim/ComfyUI';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins=json.loads(Path(__file__).with_name('source-pins.json').read_text())
for name,digest in pins.items():assert sha(source/name)==digest,name
sys.path.insert(0,str(source));sys.argv=['coordinator-original-cosmos-empty','--cpu','--fp32-text-enc','--disable-dynamic-vram','--disable-xformers']
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sim.options
sim.options.enable_args_parsing()
import torch
import sim.model_management as management
from sim_extras import nodes_cosmos
from sim_api.latest import io
torch.set_num_threads(1)
assert Path(nodes_cosmos.__file__).resolve()==source/'sim_extras/nodes_cosmos.py'
assert str(management.intermediate_device())=='cpu'
cases=[]
for class_type in ['CosmosImageToVideoLatent','CosmosPredict2ImageToVideoLatent']:
 for length in [1,4,5,8,9,81]:
  for batch_size in [1,3,64]:
   unused_vae=object()
   with torch.inference_mode():returned=getattr(nodes_cosmos,class_type).execute(unused_vae,24,16,length,batch_size)
   assert isinstance(returned,io.NodeOutput) and len(returned.args)==1
   latent=returned.args[0];assert set(latent)=={'samples'}
   samples=latent['samples'];assert samples.dtype==torch.float32 and str(samples.device)=='cpu' and torch.isfinite(samples).all().item()
   values=samples.detach().contiguous().flatten().tolist();assert all(value==0.0 for value in values)
   cases.append({'class_type':class_type,'width':24,'height':16,'length':length,'batch_size':batch_size,'start_image':'absent','end_image':'absent','latent_keys':sorted(latent),'shape':list(samples.shape),'dtype':str(samples.dtype),'values':values,'storage_sha256':hashlib.sha256(samples.detach().contiguous().numpy().tobytes()).hexdigest()})
for name,digest in pins.items():assert sha(source/name)==digest,name
value={'format':'sim.source.cosmos-empty.original-cpu-f32.v1','producer_sha256':sha(__file__),'source_sha256':pins,'torch_version':torch.__version__,'cases':cases,'case_count':len(cases),'qualification':'Genuine original imported two Cosmos classes, io.NodeOutput and real Torch CPU/F32 zeros. Absent image branches only: Source VAE argument is inert and no encode is reached. Does not prove learned VAE, image-present path, native handle admission, mapped executor or application acceptance.'}
assert len(cases)==36
out=Path(__file__).with_name('source-original-cosmos-empty.json');assert not out.exists();out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({'path':str(out),'sha256':sha(out),'cases':len(cases)}))
