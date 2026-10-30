import argparse, ast, hashlib, json, math, types
from pathlib import Path
import torch
ROOT = Path('/home/asvegah/repos/sim/projects/sim/ComfyUI')
FILES = ['nodes.py','sim_extras/nodes_latent.py','sim/utils.py','sim_extras/nodes_post_processing.py']
NAMES = ['LatentFlip','LatentFromBatch','LatentInterpolate','LatentMultiply','LatentOperationSharpen','LatentOperationTonemapReinhard','LatentRotate','LatentSubtract','LatentUpscale','LatentUpscaleBy']
torch.set_num_threads(1)
def selected(path,names,namespace):
    tree=ast.parse((ROOT/path).read_text()); body=[n for n in tree.body if getattr(n,'name',None) in names]; assert len(body)==len(names)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(ROOT/path),'exec'),namespace)
utils={'torch':torch,'math':math}; selected('sim/utils.py',['bislerp','common_upscale','repeat_to_batch_size'],utils)
post={'torch':torch}; selected('sim_extras/nodes_post_processing.py',['gaussian_kernel'],post)
sim=types.SimpleNamespace(utils=types.SimpleNamespace(**utils)); sim_extras=types.SimpleNamespace(nodes_post_processing=types.SimpleNamespace(**post))
io=types.SimpleNamespace(ComfyNode=object,NodeOutput=lambda *args: args)
namespace={'torch':torch,'math':math,'MAX_RESOLUTION':16384,'io':io,'sim':sim,'sim_extras':sim_extras,'nodes':types.SimpleNamespace(MAX_RESOLUTION=16384)}
selected('nodes.py',['LatentFlip','LatentFromBatch','LatentRotate','LatentUpscale','LatentUpscaleBy'],namespace)
selected('sim_extras/nodes_latent.py',['reshape_latent_to',*NAMES[2:6],'LatentSubtract'],namespace)
def wire_tensor(value):return {'shape':list(value.shape),'values':[v if math.isfinite(v) else str(v) for v in value.flatten().tolist()]}
def wire(latent):return {k:wire_tensor(v) if isinstance(v,torch.Tensor) else v for k,v in latent.items()}
def samples(shape=(2,2,3,5),offset=0,mask_shape=(1,3,5),indices=None,zero=False):
    tensor=torch.arange(math.prod(shape),dtype=torch.float32).reshape(shape);tensor=(tensor.remainder(17)-8)*0.125+offset
    if zero:tensor.zero_()
    result={'samples':tensor,'spatial_downscale_ratio':8,'source_marker':'preserved source dictionary'}
    if mask_shape:result['noise_mask']=torch.arange(math.prod(mask_shape),dtype=torch.float32).reshape(mask_shape)*0.1
    if indices is not None:result['batch_index']=indices
    return result
cases=[]
def case(name,args,first=None,second=None,label=''):
    first=first or samples(); result={'node':name,'label':label,'arguments':args,'first':wire(first)}; fn=namespace[name]
    try:
        if name in ['LatentOperationSharpen','LatentOperationTonemapReinhard']:
            operation=fn.execute(**args)[0]; output=dict(first);output['samples']=operation(first['samples']);result['operation']=True
        elif name in ['LatentSubtract','LatentInterpolate']:
            second=second or samples(shape=(1,2,5,3),offset=0.3);result['second']=wire(second);output=fn.execute(first,second,**args)[0]
        elif name=='LatentMultiply':output=fn.execute(first,**args)[0]
        else:output=getattr(fn(),fn.FUNCTION)(first,**args)[0]
        result['output']=wire(output);result['returns_original']=output is first
    except Exception as error:result.update(error_type=type(error).__name__,error=str(error))
    cases.append(result)
for method in ['x-axis: vertically','y-axis: horizontally']:case('LatentFlip',{'flip_method':method})
for rotation in ['none','90 degrees','180 degrees','270 degrees']:case('LatentRotate',{'rotation':rotation})
for index,length in [(-1,1),(-99,3),(1,64),(99,1),(0,2)]:
 for mask in [(1,3,5),(2,1,3,5),(1,1,3,5)]:case('LatentFromBatch',{'batch_index':index,'length':length},samples(mask_shape=mask,indices=[41,43]))
for value in [-2,0,0.125,10]:case('LatentMultiply',{'multiplier':value})
case('LatentSubtract',{})
for shape in [(2,2,3,5),(1,4,3,5),(3,2,3,5)]:
 for ratio in [0,0.35,1]:case('LatentInterpolate',{'ratio':ratio},samples(shape=shape),samples(shape=(1,shape[1],5,3),offset=0.3))
case('LatentInterpolate',{'ratio':0.5},samples(zero=True),samples(zero=True))
for method in ['nearest-exact','bilinear','area','bicubic','bislerp']:
 for dims in [(64,64),(0,72),(80,0),(0,0)]:case('LatentUpscale',{'upscale_method':method,'width':dims[0],'height':dims[1],'crop':'center'})
 for scale in [0.5,1.5,2]:case('LatentUpscaleBy',{'upscale_method':method,'scale_by':scale})
for multiplier in [0,1,2]:case('LatentOperationTonemapReinhard',{'multiplier':multiplier},samples(shape=(1,2,3,5)))
for alpha in [0,0.1,1]:case('LatentOperationSharpen',{'sharpen_radius':1,'sigma':1.0,'alpha':alpha},samples(shape=(1,2,3,5)))
case('LatentOperationSharpen',{'sharpen_radius':9,'sigma':1.0,'alpha':0.1},samples(shape=(1,2,3,5)),label='source reflect padding failure')
for indices in [None,[10,11,12,13,14]]:
    case('LatentFromBatch',{'batch_index':2,'length':3},samples(shape=(5,2,3,5),mask_shape=(2,1,3,5),indices=indices))
for indices in [[],[99],[10,11,12,13,14,15,16]]:
    case('LatentFromBatch',{'batch_index':2,'length':3},samples(shape=(5,2,3,5),mask_shape=(2,1,3,5),indices=indices),label='source simultaneous mask and short/long/empty retained-index slice')
for name,args in [('LatentFlip',{'flip_method':'x-axis: vertically'}),('LatentRotate',{'rotation':'90 degrees'}),('LatentUpscaleBy',{'upscale_method':'bilinear','scale_by':1.5})]:
    case(name,args,samples(shape=(1,2,2,3,5),mask_shape=(1,1,2,3,5)))
case('LatentFromBatch',{'batch_index':2,'length':3},samples(shape=(5,2,3,5),mask_shape=(2,3,5)),label='source four-repeat dimensions prepend rank-three mask')
case('LatentFromBatch',{'batch_index':2,'length':3},samples(shape=(5,2,2,3,5),mask_shape=(2,1,2,3,5)),label='source four-repeat dimensions reject rank-five mask')
aliases={}
for path in FILES[:2]:
    for node in ast.parse((ROOT/path).read_text()).body:
        if isinstance(node,ast.ClassDef) and node.name in NAMES:
            values=[]
            for child in ast.walk(node):
                if isinstance(child,ast.keyword) and child.arg=='search_aliases':values=ast.literal_eval(child.value)
                if isinstance(child,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SEARCH_ALIASES' for t in child.targets):values=ast.literal_eval(child.value)
            aliases[node.name]=values
result={'aliases':aliases,'source_sha256' :{path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in FILES},'torch':torch.__version__,'nodes':NAMES,'cases':cases}
content=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n';parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();path=Path(__file__).with_name('source-contract.json')
if args.check:assert path.read_text()==content,'Independent source fixture drift'
else:path.write_text(content)
print('actual Torch source',len(cases),'cases',sum('output' in c for c in cases),'success',sum('error' in c for c in cases),'errors')
