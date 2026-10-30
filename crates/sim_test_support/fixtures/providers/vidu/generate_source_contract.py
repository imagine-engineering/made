"""Independent Vidu node/API/helper oracle; native video codec and poll clock remain unbound."""
import argparse,ast,asyncio,contextlib,copy,hashlib,json,logging,math,os,sys,tempfile,time
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from types import ModuleType,SimpleNamespace
import numpy as np,torch
from PIL import Image
ROOT=Path('/home/asvegah/repos/sim/projects/comfy/ComfyUI');DEST=Path(__file__).with_name('source-contract.json')
FILES=['comfy_api_nodes/nodes_vidu.py','comfy_api_nodes/apis/vidu.py','comfy_api_nodes/util/upload_helpers.py','comfy_api_nodes/util/conversions.py','comfy_api_nodes/util/download_helpers.py','comfy_api_nodes/util/validation_utils.py','comfy/utils.py','comfy_api_nodes/util/client.py']
NODES=['Vidu2ImageToVideoNode','Vidu2ReferenceVideoNode','Vidu2StartEndToVideoNode','Vidu2TextToVideoNode','Vidu3ImageToVideoNode','Vidu3StartEndToVideoNode','Vidu3TextToVideoNode','ViduExtendVideoNode','ViduImageToVideoNode','ViduMultiFrameVideoNode','ViduReferenceVideoNode','ViduStartEndToVideoNode','ViduTextToVideoNode']
VIDEO=Path(__file__).with_name('encoded-video.mp4');torch.set_num_threads(1)
def selected(file,names,namespace,classes=False):
 nodes=[]
 for original in ast.parse((ROOT/file).read_text()).body:
  if getattr(original,'name',None) not in names:continue
  node=copy.deepcopy(original)
  if classes:node.body=[member for member in node.body if not isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)) or member.name!='define_schema']
  for member in ast.walk(node):
   if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
    member.returns=None
    for argument in [*member.args.args,*member.args.kwonlyargs]:argument.annotation=None
  nodes.append(node)
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(ROOT/file),'exec'),namespace)
async def generate():
 requests=[];constructors=[];polls=[];uploads=[];downloads=[];conversions=[];events=[];counts=[];options={};uuid_number=0
 class Container(Enum):MP4='mp4'
 class Codec(Enum):H264='h264'
 class VideoBoundary:
  def __init__(self,descriptor):self.descriptor=descriptor
  def get_duration(self):
   if self.descriptor.get('duration_error'):raise ValueError('source duration unavailable')
   return self.descriptor['duration']
  def save_to(self,dest,format,codec):
   conversions.append({'format':format.value,'codec':codec.value});events.append({'convert':'video'})
   if self.descriptor.get('conversion_error'):raise ValueError('source conversion unavailable')
   dest.write(VIDEO.read_bytes())
 class DownloadBoundary:
  def __init__(self,stream):self.stream=stream
 namespace={'__builtins__':__builtins__,'torch':torch,'np':np,'Image':Image,'BytesIO':BytesIO,'Path':Path,'math':math,'os':os,'time':time,'logging':logging,'IO':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *values:values),'Input':SimpleNamespace(),'Types':SimpleNamespace(VideoContainer=Container,VideoCodec=Codec),'InputImpl':SimpleNamespace(VideoFromFile=DownloadBoundary),'mimetype_to_extension':lambda mime:mime.split('/')[-1].lower()}
 api=ModuleType('pinned_vidu_api');sys.modules[api.__name__]=api;exec(compile((ROOT/FILES[1]).read_text(),str(ROOT/FILES[1]),'exec'),api.__dict__);namespace.update(api.__dict__)
 def uuid4():
  nonlocal uuid_number
  uuid_number+=1;return f'00000000-0000-4000-8000-{uuid_number:012d}'
 namespace['uuid']=SimpleNamespace(uuid4=uuid4)
 selected(FILES[6],{'common_upscale'},namespace)
 selected(FILES[3],{'tensor_to_bytesio','tensor_to_pil','pil_to_bytesio','downscale_image_tensor','_compute_downscale_dims'},namespace)
 selected(FILES[5],{'get_image_dimensions','get_number_of_images','validate_image_dimensions','validate_image_aspect_ratio','validate_images_aspect_ratio_closeness','validate_string','validate_video_duration','_ratio_from_tuple','_assert_ratio_bounds'},namespace)
 async def upload(cls,stream,filename,mime,wait_label='Uploading',progress_origin_ts=None):
  index=len(uploads);uploads.append({'file_name':filename,'mime_type':mime,'wait_label':wait_label,'bytes_hex':stream.getvalue().hex()});events.append({'upload':index});return f'https://source.fixture.invalid/input-{index+1}.bin'
 namespace['upload_file_to_comfyapi']=upload
 selected(FILES[2],{'upload_images_to_comfyapi','upload_image_to_comfyapi','upload_video_to_comfyapi'},namespace)
 async def sync(cls,endpoint,response_model,data=None,**kwargs):
  payload={'task_id':options.get('task_id','source-task-42'),'state':options.get('creation_state','created'),'created_at':'2026-09-30T00:00:00Z','code':options.get('code',None)}
  if options.get('missing_task_id'):payload.pop('task_id')
  record={'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(mode='json',exclude_none=True),'options':kwargs,'response':copy.deepcopy(payload)};requests.append(record);events.append({'request':endpoint.path})
  response=response_model.model_validate(payload);record['normalized_response']=response.model_dump(mode='json',exclude_none=True);return response
 async def poll(cls,endpoint,response_model,**kwargs):
  record={'path':endpoint.path,'method':endpoint.method,'options':{key:normalize(value) for key,value in kwargs.items() if not callable(value)},'responses':[],'statuses':[],'progress_updates':[]};polls.append(record);events.append({'poll':endpoint.path})
  queue=[]
  for state,progress in [('queueing',0),('processing',.5),(options.get('poll_state','success'),1)]:
   queue.append({'state':state,'err_code':options.get('err_code',None),'progress':options.get('progress',progress),'credits':options.get('credits',7),'creations':options.get('creations',[{'id':'creation-1','url':'https://source.fixture.invalid/result.mp4','cover_url':'https://source.fixture.invalid/cover.png'},{'id':'creation-2','url':'https://source.fixture.invalid/unused.mp4','cover_url':'https://source.fixture.invalid/cover2.png'}])})
  async def sync_poll(*args,**kw):
   if not queue:raise RuntimeError('oracle poll script exhausted')
   payload=queue.pop(0);record['responses'].append({'raw':copy.deepcopy(payload)})
   try:response=response_model.model_validate(payload)
   except Exception:return payload
   record['responses'][-1]['normalized']=response.model_dump(mode='json',exclude_none=True)
   record['statuses'].append({'state':kwargs['status_extractor'](response),'progress':kwargs['progress_extractor'](response),'price':kwargs['price_extractor'](response)})
   return payload
  async def virtual_delay(*args,**kw):pass
  class ProgressBar:
   def __init__(self,total):pass
   def update_absolute(self,current,total):record['progress_updates'].append({'current':current,'total':total})
  class Interrupted(Exception):pass
  class Local(Exception):pass
  class ApiError(Exception):pass
  client=dict(namespace);client.update(asyncio=asyncio,contextlib=contextlib,json=json,dataclass=dataclass,int=int,utils=SimpleNamespace(ProgressBar=ProgressBar),ProcessingInterrupted=Interrupted,LocalNetworkError=Local,ApiServerError=ApiError,is_processing_interrupted=lambda:False,sleep_with_interrupt=virtual_delay,sync_op_raw=sync_poll,_display_time_progress=lambda *args,**kwargs:None)
  for assignment in ast.parse((ROOT/FILES[7]).read_text()).body:
   if isinstance(assignment,ast.Assign):
    for target in assignment.targets:
     if isinstance(target,ast.Name) and target.id in ['COMPLETED_STATUSES','FAILED_STATUSES','QUEUED_STATUSES']:client[target.id]=ast.literal_eval(assignment.value)
  selected(FILES[7],{'_PollUIState','_validate_or_raise','_wrap_model_extractor','_normalize_statuses','_normalize_status_value','poll_op','poll_op_raw'},client)
  return await client['poll_op'](cls,endpoint,response_model=response_model,**kwargs)
 async def download(url,dest,**kwargs):downloads.append({'url':url,'options':kwargs,'bytes_hex':VIDEO.read_bytes().hex()});events.append({'download':url});dest.write(VIDEO.read_bytes());dest.seek(0)
 namespace.update(sync_op=sync,poll_op=poll,download_url_to_bytesio=download,ApiEndpoint=lambda path,method='GET':SimpleNamespace(path=path,method=method))
 selected(FILES[4],{'download_url_to_video_output'},namespace)
 selected(FILES[0],{'execute_task'},namespace)
 selected(FILES[0],set(NODES),namespace,True)
 for assignment in ast.parse((ROOT/FILES[0]).read_text()).body:
  if isinstance(assignment,ast.Assign):
   for target in assignment.targets:
    if isinstance(target,ast.Name) and target.id.startswith('VIDU_'):namespace[target.id]=ast.literal_eval(assignment.value)
 def normalize(value):
  if isinstance(value,Enum):return value.value
  if hasattr(value,'model_dump'):return value.model_dump(mode='json',exclude_none=False)
  if isinstance(value,DownloadBoundary):return {'bytes_hex':value.stream.getvalue().hex(),'owner':'VideoFromFile boundary; no PyAV decode'}
  if isinstance(value,(tuple,list)):return [normalize(item) for item in value]
  if isinstance(value,dict):return {key:normalize(item) for key,item in value.items()}
  return value
 for name in ['TaskCreationRequest','TaskExtendCreationRequest','TaskMultiFrameCreationRequest','SubjectReference','FrameSetting']:
  original=namespace[name]
  def constructor(*args,_name=name,_original=original,**kwargs):
   record={'model':_name,'raw':normalize(kwargs)};constructors.append(record);events.append({'constructor':len(constructors)-1})
   result=_original(*args,**kwargs);record['normalized']=result.model_dump(mode='json',exclude_none=True);return result
  namespace[name]=constructor
 def source_int(raw):
  record={'raw':raw};counts.append(record);result=int(raw);record['resolved']=result;return result
 namespace['int']=source_int
 def image(batch=1,width=128,height=128,channels=3):return dict(batch=batch,width=width,height=height,channels=channels)
 def base(node):
  settings={'model':'viduq2','prompt':' source scene ','seed':42,'duration':5,'resolution':'720p','aspect_ratio':'16:9','movement_amplitude':'auto'}
  if node=='Vidu2TextToVideoNode':settings.pop('movement_amplitude');settings['background_music']=False
  if node.startswith('Vidu3'):
   settings={'model':{'model':'viduq3-pro','duration':5,'resolution':'720p','audio':True,'aspect_ratio':'16:9'},'prompt':' source scene ','seed':42}
  images={};subjects=[];video=None
  if 'ImageToVideo' in node:images['image']=image()
  if 'StartEnd' in node:images.update(first_frame=image(),end_frame=image(channels=4))
  if node=='ViduReferenceVideoNode':images['images']=image(batch=2)
  if node=='Vidu2ReferenceVideoNode':settings['audio']=True;subjects=[{'id':'second-subject','image':image(batch=2)},{'id':'first-subject','image':image()}]
  if node=='ViduExtendVideoNode':
   settings={'model':{'model':'viduq2-pro','duration':3,'resolution':'720p'},'prompt':' source scene ','seed':42};video={'duration':8,'width':128,'height':128,'container':'mp4'}
  if node=='ViduMultiFrameVideoNode':
   settings={'model':'viduq2-pro','seed':42,'resolution':'720p','frames':{'frames':'2','prompt1':' first ','duration1':3,'prompt2':' second ','duration2':4}};images.update(start_image=image(),end_image1=image(),end_image2=image(channels=4))
  return {'node':node,'settings':settings,'images':images,'subjects':subjects,'video':video}
 def tensor(descriptor,ordinal):
  image=torch.empty((descriptor['batch'],descriptor['height'],descriptor['width'],descriptor['channels']),dtype=torch.float32)
  for batch in range(descriptor['batch']):
   for channel in range(descriptor['channels']):image[batch,...,channel]=[.5,.25,.125,1.][channel]+((ordinal+batch)*.01 if channel<3 else 0.)
  return image
 def arguments(value):
  result=copy.deepcopy(value['settings']);node=value['node']
  for ordinal,(name,descriptor) in enumerate(value['images'].items()):
   item=None if descriptor is None else tensor(descriptor,ordinal)
   if node=='ViduMultiFrameVideoNode' and name.startswith('end_image'):result['frames'][name]=item
   else:result[name]=item
  if node=='Vidu2ReferenceVideoNode':result['subjects']={item['id']:tensor(item['image'],ordinal) for ordinal,item in enumerate(value['subjects'])}
  if node=='ViduExtendVideoNode':result['video']=VideoBoundary(value['video'])
  if node in ['ViduImageToVideoNode','Vidu2ImageToVideoNode','ViduStartEndToVideoNode','Vidu2StartEndToVideoNode']:result.pop('aspect_ratio',None)
  if node in ['Vidu2ImageToVideoNode','Vidu2ReferenceVideoNode','Vidu2StartEndToVideoNode']:result.pop('background_music',None)
  return result
 cases=[];errors=[]
 async def observe(value):
  nonlocal uuid_number
  uuid_number=0
  for collection in [requests,constructors,polls,uploads,downloads,conversions,events,counts]:collection.clear()
  options.clear();options.update(value.get('oracle_options',{}))
  try:result=await namespace[value['node']].execute(**arguments(value))
  except (NameError,ImportError) as error:raise RuntimeError(f'harness failure {value["node"]}: {error}') from error
  except Exception as error:errors.append({'input':copy.deepcopy(value),'error_type':type(error).__name__,'error_message':str(error),'requests':copy.deepcopy(requests),'constructors':copy.deepcopy(constructors),'polls':copy.deepcopy(polls),'uploads':copy.deepcopy(uploads),'downloads':copy.deepcopy(downloads),'conversions':copy.deepcopy(conversions),'events':copy.deepcopy(events),'counts':copy.deepcopy(counts)});return
  cases.append({'input':copy.deepcopy(value),'requests':copy.deepcopy(requests),'constructors':copy.deepcopy(constructors),'polls':copy.deepcopy(polls),'uploads':copy.deepcopy(uploads),'downloads':copy.deepcopy(downloads),'conversions':copy.deepcopy(conversions),'events':copy.deepcopy(events),'counts':copy.deepcopy(counts),'output':normalize(result)})
 for node in NODES:await observe(base(node))
 for node in NODES:
  if node!='ViduMultiFrameVideoNode':
   for prompt in ['', ' ', '\x1c padded \x1f','x'*2000,'x'*2001]:value=base(node);value['settings']['prompt']=prompt;await observe(value)
  for seed in [-1,0,2147483647,2147483648]:value=base(node);value['settings']['seed']=seed;await observe(value)
  for option in [{'creation_state':'failed','code':17},{'missing_task_id':True},{'creations':[]},{'credits':0},{'credits':None},{'credits':-1},{'poll_state':'failed','creations':[]}]:value=base(node);value['oracle_options']=option;await observe(value)
 for node in [item for item in NODES if 'ImageToVideo' in item]:
  for batch,width,height in [(0,128,128),(2,128,128),(1,32,128),(1,128,32),(1,33,128)]:value=base(node);value['images']['image']=image(batch,width,height);await observe(value)
 for node in [item for item in NODES if 'StartEnd' in item]:
  for batch,first_width,end_width in [(0,128,128),(2,128,128),(1,128,160),(1,128,161)]:value=base(node);value['images']['first_frame']=image(batch,first_width);value['images']['end_frame']=image(1,end_width);await observe(value)
 for batch,width in [(0,128),(7,128),(8,128),(1,127),(2,512)]:value=base('ViduReferenceVideoNode');value['images']['images']=image(batch,width);await observe(value)
 for subject_counts in [[],[0],[3,3,1],[3,3,2],[4],[1,0,1]]:value=base('Vidu2ReferenceVideoNode');value['subjects']=[{'id':f'subject-{ordinal}','image':image(batch)} for ordinal,batch in enumerate(subject_counts)];await observe(value)
 for duration in [3.9998,3.9999,4,55,55.0001,55.0002]:value=base('ViduExtendVideoNode');value['video']['duration']=duration;await observe(value)
 for mode in ['duration_error','conversion_error']:value=base('ViduExtendVideoNode');value['video'][mode]=True;value['images']['end_frame']=image();await observe(value)
 value=base('ViduExtendVideoNode');value['images']['end_frame']=image();await observe(value)
 value=base('ViduExtendVideoNode');value['settings']['model'].pop('resolution');value['images']['end_frame']=image();await observe(value)
 for count in ['0','1','2','2.0','invalid',-1,2.9]:value=base('ViduMultiFrameVideoNode');value['settings']['frames']['frames']=count;await observe(value)
 for field,bad in [('prompt2','x'*2001),('duration2','bad'),('duration1',2.5)]:value=base('ViduMultiFrameVideoNode');value['settings']['frames'][field]=bad;await observe(value)
 value=base('ViduMultiFrameVideoNode');value['images']['end_image2']=image(width=512);await observe(value)
 schema={}
 for original in ast.parse((ROOT/FILES[0]).read_text()).body:
  if isinstance(original,ast.ClassDef) and original.name in NODES:schema[original.name]=[ast.unparse(member) for member in original.body if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)) and member.name=='define_schema']
 return {'sources':{file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in FILES},'asset_sha256':hashlib.sha256(VIDEO.read_bytes()).hexdigest(),'versions':{'torch':torch.__version__,'pillow':Image.__version__},'cases':cases,'errors':errors,'schema_facts':schema,'scope':'actual node/API/validation/PNG upload/client poll control+extractors and video/download helper boundary; retained real MP4 stand-in for Video.save_to and VideoFromFile, virtual delay/UI boundary, no PyAV codec/real clock/native ABI proof'}
parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();result=asyncio.run(generate());encoded=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
if args.check:
 if DEST.read_text()!=encoded:raise SystemExit('source oracle drift')
else:DEST.write_text(encoded)
print(f"Vidu actual source boundary: {len(result['cases'])} successes,{len(result['errors'])} ordered failures")
