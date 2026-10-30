"""Pinned Runway execution/API/upload helper oracle; native video ownership is separate."""
import argparse,ast,asyncio,copy,hashlib,inspect,json,logging,math,sys,time,uuid
from enum import Enum
from io import BytesIO
from pathlib import Path
from types import ModuleType,SimpleNamespace
import numpy as np,torch
from PIL import Image
ROOT=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI');DEST=Path(__file__).with_name('source-contract.json')
FILES=['sim_api_nodes/nodes_runway.py','sim_api_nodes/apis/runway.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/upload_helpers.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/download_helpers.py','sim_api_nodes/util/client.py','sim/utils.py','sim_api/latest/_input_impl/video_types.py']
NODES=['RunwayAleph2KeyframeNode','RunwayAleph2PromptImageNode','RunwayAleph2VideoToVideoNode','RunwayFirstLastFrameNode','RunwayImageToVideoNodeGen3a','RunwayImageToVideoNodeGen4','RunwayTextToImageNode'];torch.set_num_threads(1)
def selected(path,names,namespace,classes=False):
 output=[]
 for original in ast.parse((ROOT/path).read_text()).body:
  if getattr(original,'name',None) not in names:continue
  node=copy.deepcopy(original)
  if classes:node.body=[member for member in node.body if not isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)) or member.name!='define_schema']
  for member in ast.walk(node):
   if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
    member.returns=None
    for argument in [*member.args.args,*member.args.kwonlyargs]:argument.annotation=None
  output.append(node)
 exec(compile(ast.fix_missing_locations(ast.Module(body=output,type_ignores=[])),str(ROOT/path),'exec'),namespace)
def image(value):
 result=torch.empty((value.get('batch',1),value.get('height',2),value.get('width',2),value.get('channels',3)),dtype=torch.float32)
 for index in range(result.shape[0]):
  for channel in range(result.shape[-1]):result[index,...,channel]=[.5,.25,.125,1.][channel]+(index*.01 if channel<3 else 0)
 return result
async def generate():
 requests=[];polls=[];uploads=[];downloads=[];events=[];options={};namespace={'__builtins__':__builtins__,'torch':torch,'np':np,'Image':Image,'BytesIO':BytesIO,'math':math,'Enum':Enum,'time':time,'logging':logging,'IO':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *values:values),'Input':SimpleNamespace(),'uuid':SimpleNamespace(uuid4=lambda:'00000000-0000-4000-8000-000000000000'),'mimetype_to_extension':lambda mime:mime.split('/')[-1].lower()}
 api=ModuleType('pinned_runway_api');sys.modules[api.__name__]=api;exec(compile((ROOT/FILES[1]).read_text(),str(ROOT/FILES[1]),'exec'),api.__dict__);namespace.update(api.__dict__)
 namespace.update(Model=namespace['RunwayModelEnum'],Duration=namespace['RunwayDurationEnum'],AspectRatio=namespace['RunwayAspectRatioEnum'],TaskStatusResponse=namespace['RunwayTaskStatusResponse'])
 for assignment in ast.parse((ROOT/FILES[0]).read_text()).body:
  if isinstance(assignment,ast.Assign):
   try:value=ast.literal_eval(assignment.value)
   except (ValueError,TypeError):continue
   for target in assignment.targets:
    if isinstance(target,ast.Name):namespace[target.id]=value
 selected(FILES[7],{'common_upscale'},namespace)
 selected(FILES[2],{'image_tensor_pair_to_batch','tensor_to_bytesio','tensor_to_pil','pil_to_bytesio','downscale_image_tensor','_compute_downscale_dims','bytesio_to_image_tensor'},namespace)
 selected(FILES[4],{'get_image_dimensions','validate_string','validate_image_dimensions','validate_image_aspect_ratio','_assert_ratio_bounds','_ratio_from_tuple','validate_video_duration'},namespace)
 class VideoContainer(str,Enum):MP4='mp4'
 class VideoCodec(str,Enum):H264='h264'
 namespace['Types']=SimpleNamespace(VideoContainer=VideoContainer,VideoCodec=VideoCodec)
 encoded_video=Path(__file__).with_name('encoded-video.mp4').read_bytes()
 class VideoOwnerProbe:
  def __init__(self,value):self.value=value
  def get_duration(self):
   if self.value.get('duration_error'):raise RuntimeError('source metadata probe failure')
   return self.value['duration']
  def get_frame_rate(self):
   if self.value.get('rate_error'):raise RuntimeError('source rate probe failure')
   return self.value['fps']
  def save_to(self,stream,format,codec):
   events.append({'video_save':{'format':format.value,'codec':codec.value}})
   if self.value.get('save_error'):raise RuntimeError('source native video owner unavailable')
   stream.write(encoded_video)
 async def upload_file(cls,stream,filename,mime,wait_label='Uploading',progress_origin_ts=None):
  record={'file_name':filename,'mime_type':mime,'bytes_hex':stream.getvalue().hex(),'wait_label':wait_label};uploads.append(record);events.append({'upload':len(uploads)-1});return f'https://source.fixture.invalid/upload-{len(uploads)}.bin'
 namespace['upload_file_to_simapi']=upload_file
 selected(FILES[3],{'upload_images_to_simapi','upload_image_to_simapi','upload_video_to_simapi'},namespace)
 async def sync(cls,endpoint,response_model,data=None,**kwargs):
  requests.append({'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(mode='json',exclude_none=True),'options':kwargs});events.append({'request':endpoint.path})
  return response_model.model_validate({'id':None if options.get('missing_id') else 'task-42'})
 async def poll(cls,endpoint,response_model,**kwargs):
  record={'path':endpoint.path,'method':endpoint.method,'estimated_duration':kwargs.get('estimated_duration'),'statuses':[]}
  for status,progress in [('PENDING',None),('RUNNING',.25),('SUCCEEDED',1)]:
   response=response_model.model_validate({'createdAt':'2026-10-01T00:00:00Z','id':'task-42','status':status,'progress':progress,'output':[] if options.get('empty_output') else ['https://source.fixture.invalid/result-first.bin','https://source.fixture.invalid/result-ignored.bin']})
   record['statuses'].append({'status':kwargs['status_extractor'](response),'progress':kwargs['progress_extractor'](response)})
  polls.append(record);events.append({'poll':endpoint.path});return response
 stream=BytesIO();Image.new('RGBA',(2,2),(128,64,32,128)).save(stream,format='PNG');png=stream.getvalue()
 async def download_image(url):
  downloads.append({'url':url,'kind':'image','bytes_hex':png.hex()});events.append({'download':url});return namespace['bytesio_to_image_tensor'](BytesIO(png))
 async def download_video(url):
  downloads.append({'url':url,'kind':'video','bytes_hex':encoded_video.hex()});events.append({'download':url});return {'video_owner_url':url}
 namespace.update(sync_op=sync,poll_op=poll,download_url_to_image_tensor=download_image,download_url_to_video_output=download_video,ApiEndpoint=lambda path,method='GET':SimpleNamespace(path=path,method=method))
 selected(FILES[0],{'RunwayGen4TurboAspectRatio','RunwayGen3aAspectRatio','get_video_url_from_task_status','get_image_url_from_task_status','get_response','generate_video',*NODES},namespace,True)
 def normalize(value):
  if isinstance(value,torch.Tensor):return {'shape':list(value.shape),'pixels':value.flatten().tolist()}
  if isinstance(value,(namespace['RunwayAleph2KeyframeChain'],namespace['RunwayAleph2PromptImageChain'])):
   return {'kind':'keyframe' if isinstance(value,namespace['RunwayAleph2KeyframeChain']) else 'prompt_image','items':[{'image':normalize(item.image),'mode':item.mode,'value':item.value} for item in value.items]}
  if isinstance(value,(tuple,list)):return [normalize(item) for item in value]
  return value
 def base(node):
  value={'node':node,'settings':{'prompt':' ☁ ','duration':5,'ratio':'1280:720','seed':7,'public_figure_threshold':'low'},'images':[],'video':None,'keyframes':None,'prompt_images':None}
  if node in {'RunwayImageToVideoNodeGen3a','RunwayImageToVideoNodeGen4','RunwayFirstLastFrameNode'}:
   value['images']=[{'role':'start','batch':1,'width':2,'height':2}];value['settings']['ratio']='1280:768' if node!='RunwayImageToVideoNodeGen4' else '1280:720'
  if node=='RunwayFirstLastFrameNode':value['images'].append({'role':'end','batch':1,'width':2,'height':2})
  if node=='RunwayTextToImageNode':value['settings']['ratio']='1920:1080'
  if node in {'RunwayAleph2KeyframeNode','RunwayAleph2PromptImageNode'}:
   value['images']=[{'role':'image','batch':1,'width':2,'height':2}];value['settings']['timing' if 'Keyframe' in node else 'position']={'timing' if 'Keyframe' in node else 'position':namespace['_TIMING_ABSOLUTE'],'seconds':1.,'fraction':.5}
  if node=='RunwayAleph2VideoToVideoNode':value['video']={'duration':5.,'fps':24.}
  return value
 def chain(entries,keyframes):
  result=namespace['RunwayAleph2KeyframeChain' if keyframes else 'RunwayAleph2PromptImageChain']()
  for item in entries:result.add(namespace['RunwayAleph2KeyframeItem' if keyframes else 'RunwayAleph2PromptImageItem'](image(item['image']),item['mode'],item['value']))
  return result
 def arguments(value):
  method=namespace[value['node']].execute;parameters=inspect.signature(method).parameters;output={key:copy.deepcopy(val) for key,val in value['settings'].items() if key in parameters}
  for role,key in [('start','start_frame'),('end','end_frame'),('image','image'),('reference','reference_image')]:
   found=next((entry for entry in value['images'] if entry['role']==role),None)
   if found is not None:output[key]=image(found)
  if value['video'] is not None:output['video']=VideoOwnerProbe(value['video'])
  if value['keyframes'] is not None:output['keyframes']=chain(value['keyframes'],True)
  if value['prompt_images'] is not None:output['prompt_images']=chain(value['prompt_images'],False)
  return output
 cases=[];errors=[]
 async def observe(value):
  for output in [requests,polls,uploads,downloads,events]:output.clear()
  options.clear();options.update(value.get('oracle_options',{}))
  try:
   result=namespace[value['node']].execute(**arguments(value));result=await result if asyncio.iscoroutine(result) else result
  except (NameError,ImportError,AttributeError) as error:raise RuntimeError(f'oracle harness failure: {value["node"]}: {error}') from error
  except Exception as error:errors.append({'input':copy.deepcopy(value),'error_type':type(error).__name__,'requests':copy.deepcopy(requests),'polls':copy.deepcopy(polls),'uploads':copy.deepcopy(uploads),'downloads':copy.deepcopy(downloads),'events':copy.deepcopy(events)});return
  cases.append({'input':copy.deepcopy(value),'requests':copy.deepcopy(requests),'polls':copy.deepcopy(polls),'uploads':copy.deepcopy(uploads),'downloads':copy.deepcopy(downloads),'events':copy.deepcopy(events),'output':normalize(result)})
 for node in NODES:await observe(base(node))
 schema_facts={};price_badges={}
 for cls in ast.parse((ROOT/FILES[0]).read_text()).body:
  if not isinstance(cls,ast.ClassDef) or cls.name not in NODES:continue
  schema=next(member for member in cls.body if isinstance(member,ast.FunctionDef) and member.name=='define_schema');schema_facts[cls.name]=ast.unparse(schema)
  for call in ast.walk(schema):
   if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='PriceBadge':price_badges[cls.name]=next(keyword.value.value for keyword in call.keywords if keyword.arg=='expr')
 for node in NODES:
  if node in {'RunwayAleph2KeyframeNode','RunwayAleph2PromptImageNode'}:
   key='timing' if 'Keyframe' in node else 'position'
   for mode in [namespace['_TIMING_ABSOLUTE'],namespace['_TIMING_FRACTION'],'unknown']:
    for number in [0.,.5,1.,30.]:value=base(node);value['settings'][key]={key:mode,'seconds':number,'fraction':number};await observe(value)
   value=base(node);value['keyframes' if 'Keyframe' in node else 'prompt_images']=[{'image':{'batch':1,'width':2,'height':2},'mode':'seconds' if 'Keyframe' in node else 'timestamp','value':.5}];await observe(value)
  else:
   for prompt in ['', ' ', '☁'*1001,' '+'☁'*1000+' ']:value=base(node);value['settings']['prompt']=prompt;await observe(value)
   for option in [{'empty_output':True},{'missing_id':True}]:value=base(node);value['oracle_options']=option;await observe(value)
 for node in ['RunwayImageToVideoNodeGen3a','RunwayImageToVideoNodeGen4','RunwayFirstLastFrameNode']:
  for ratio in namespace['RunwayGen3aAspectRatio'] if node!='RunwayImageToVideoNodeGen4' else namespace['RunwayGen4TurboAspectRatio']:
   for duration in [5,10]:value=base(node);value['settings'].update(ratio=ratio.value,duration=duration);await observe(value)
  for key,number in [('duration','5'),('duration',6),('seed',-1),('seed',4294967296),('ratio','bad')]:value=base(node);value['settings'][key]=number;await observe(value)
  for width,height,batch in [(2,4,1),(4,2,1),(3,2,1),(2,3,1),(8000,2,1),(2,2,0),(2,2,2)]:value=base(node);value['images'][0].update(width=width,height=height,batch=batch);await observe(value)
 value=base('RunwayFirstLastFrameNode');value['images'][1].update(width=3,height=2);await observe(value)
 for ratio in namespace['RunwayTextToImageAspectRatioEnum']:
  for reference in [False,True]:value=base('RunwayTextToImageNode');value['settings']['ratio']=ratio.value;value['images']=[{'role':'reference','width':2,'height':2,'batch':1}] if reference else [];await observe(value)
 for duration in [1.9998,1.9999,2.,30.,30.0001,30.0002]:value=base('RunwayAleph2VideoToVideoNode');value['video']['duration']=duration;await observe(value)
 for fps in [30.,30.01,30.0101]:value=base('RunwayAleph2VideoToVideoNode');value['video']['fps']=fps;await observe(value)
 for errors_kind in ['duration_error','rate_error','save_error']:value=base('RunwayAleph2VideoToVideoNode');value['video'][errors_kind]=True;await observe(value)
 for field,mode in [('keyframes','seconds'),('keyframes','at'),('prompt_images','timestamp'),('prompt_images','position')]:
  for count in [0,1,5,6]:
   for number in [-.1,0.,.5,1.,5.0001,5.0002]:value=base('RunwayAleph2VideoToVideoNode');value[field]=[{'image':{'batch':1,'width':2,'height':2},'mode':mode,'value':number} for _ in range(count)];await observe(value)
 value=base('RunwayAleph2VideoToVideoNode');value['keyframes']=[{'image':{'batch':1,'width':2,'height':2},'mode':'seconds','value':.5}];value['prompt_images']=[{'image':{'batch':1,'width':2,'height':2},'mode':'timestamp','value':.5}];await observe(value)
 api_model_cases=[]
 for payload in [
  {'createdAt':'2026-10-01T00:00:00Z','id':'task-42','status':'PENDING'},
  {'createdAt':'2026-10-01T00:00:00Z','id':'task-42','status':'RUNNING','progress':'0.25','output':None},
  {'createdAt':1790812800,'id':'task-42','status':'RUNNING','progress':True},
  {'createdAt':'not a datetime','id':'task-42','status':'PENDING'},
  {'createdAt':'2026-10-01T00:00:00Z','id':'task-42','status':'PENDING','output':[123]},
  {'createdAt':'2026-10-01T00:00:00Z','id':'task-42','status':'RUNNING','progress':1.1},
  {'createdAt':'2026-10-01T00:00:00Z','id':'task-42','status':'RUNNING','progress':-.1},
  {'id':'task-42','status':'PENDING'},
 ]:
  try:api_model_cases.append({'payload':payload,'normalized':namespace['TaskStatusResponse'].model_validate(payload).model_dump(mode='json',exclude_none=True)})
  except Exception as error:api_model_cases.append({'payload':payload,'error_type':type(error).__name__})
 return {'api_model_cases':api_model_cases,'sources':{file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in FILES},'versions':{'torch':torch.__version__,'pillow':Image.__version__},'cases':cases,'errors':errors,'schema_facts':schema_facts,'price_badges':price_badges,'encoded_video_sha256':hashlib.sha256(encoded_video).hexdigest(),'scope':'actual node/API/validation/image conversion/upload helpers; structural video owner probes with real existing MP4 bytes; native PyAV/video decode/Model/guest separate and unclaimed'}
parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();result=asyncio.run(generate());encoded=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
if args.check:
 if DEST.read_text()!=encoded:raise SystemExit('source oracle drift')
else:DEST.write_text(encoded)
print(f"Runway actual source boundary: {len(result['cases'])} successes,{len(result['errors'])} ordered failures; native PyAV separate")
