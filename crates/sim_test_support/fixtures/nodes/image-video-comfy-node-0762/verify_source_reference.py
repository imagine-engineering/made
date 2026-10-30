import ast,hashlib,json,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
class Tensor(np.ndarray):
    @property
    def device(self):return 'cpu'
def tensor(data,dtype=None,device=None):return np.asarray(data,dtype=dtype).view(Tensor)
source_path=Path('projects/sim/ComfyUI/sim_extras/nodes_wandancer.py')
expected=json.loads(Path(__file__).with_name('fixture.json').read_text())['padding_reference']
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == expected['source_sha256']
module=ast.parse(source_path.read_text())
nodes=[node for node in module.body if isinstance(node,ast.ClassDef) and node.name in {'WanDancerPadKeyframes','WanDancerPadKeyframesList'}]
assert len(nodes)==2
namespace={'math':math,'torch':SimpleNamespace(zeros=lambda shape,dtype,device:np.zeros(shape,dtype=dtype).view(Tensor),tensor=tensor,long=np.int64),'io':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *values:values)}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source_path),'exec'),namespace)
cases=[]
for image_count,audio_frames,sample_rate,length,index in [(0,100,100,5,0),(1,0,100,5,0),(2,20,100,5,0),(2,21,100,5,0),(4,100,100,5,0),(4,100,100,5,1),(4,100,100,5,5),(4,100,100,5,6),(4,100,100,5,100),(2,100,100,1,0),(2,100,100,1,1),(10,300,100,7,2),(3,999,44100,149,0),(9,44100,44100,149,0),(7,22222,44100,13,1),(100,100,100,5,0),(100,100,100,5,4),(3,0,100,10000,100),(2,44101,44100,1,100)]:
 images=np.empty((image_count,2,3,3),dtype=np.float32).view(Tensor)
 for frame in range(image_count):images[frame]=frame+1
 waveform=tensor(np.arange(audio_frames,dtype=np.float32).reshape(1,1,audio_frames))
 output=namespace['WanDancerPadKeyframes'].execute(images,length,index,{'waveform':waveform,'sample_rate':sample_rate})
 keyframes,mask,audio=output
 samples=audio['waveform'].reshape(-1)
 cases.append({'images':image_count,'audio_frames':audio_frames,'sample_rate':sample_rate,'segment_length':length,'segment_index':index,'keyframe_values':keyframes[:,0,0,0].tolist(),'mask_values':mask[:,0,0].tolist(),'audio_length':len(samples),'audio_first':float(samples[0]) if len(samples) else None,'audio_last':float(samples[-1]) if len(samples) else None})
assert expected == {'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),'cases':cases}
print('Verified',len(cases),'pinned-source padding cases')
