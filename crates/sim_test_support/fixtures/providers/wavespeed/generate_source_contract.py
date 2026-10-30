"""Execute pinned Wavespeed nodes/API/helpers/client; video codec and real clock stay unbound."""
import argparse,re,ast,asyncio,contextlib,copy,hashlib,json,logging,math,os,sys,tempfile,time
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from types import ModuleType,SimpleNamespace
import numpy as np,torch
from PIL import Image
ROOT=Path('/home/asvegah/repos/sim/projects/comfy/ComfyUI');DEST=Path(__file__).with_name('source-contract.json')
FILES=['comfy_api_nodes/nodes_wavespeed.py','comfy_api_nodes/apis/wavespeed.py','comfy_api_nodes/util/upload_helpers.py','comfy_api_nodes/util/conversions.py','comfy_api_nodes/util/download_helpers.py','comfy_api_nodes/util/validation_utils.py','comfy/utils.py','comfy_api_nodes/util/client.py']
NODES=['WavespeedFlashVSRNode','WavespeedImageUpscaleNode'];VIDEO=Path(__file__).with_name('encoded-video.mp4');IMAGE=Path(__file__).with_name('downloaded-image.png');torch.set_num_threads(1)
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
 requests=[];constructors=[];polls=[];uploads=[];downloads=[];conversions=[];events=[];lowers=[];options={};uuid_number=0
 class Container(Enum):MP4='mp4'
 class Codec(Enum):H264='h264'
 class VideoBoundary:
  def __init__(self,descriptor):self.descriptor=descriptor
  def get_container_format(self):
   events.append({'container':True})
   if self.descriptor.get('container_error'):raise ValueError('source container unavailable')
   return self.descriptor['container']
  def get_duration(self):
   events.append({'duration':True})
   if self.descriptor.get('duration_error'):raise ValueError('source duration unavailable')
   return self.descriptor['duration']
  def save_to(self,dest,format,codec):
   conversions.append({'format':format.value,'codec':codec.value});events.append({'convert':'video'})
   if self.descriptor.get('conversion_error'):raise ValueError('source conversion unavailable')
   dest.write(VIDEO.read_bytes())
 class DownloadBoundary:
  def __init__(self,stream):self.stream=stream
 namespace={'__builtins__':__builtins__,'torch':torch,'np':np,'Image':Image,'BytesIO':BytesIO,'Path':Path,'math':math,'os':os,'time':time,'logging':logging,'IO':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *values:values),'Input':SimpleNamespace(),'Types':SimpleNamespace(VideoContainer=Container,VideoCodec=Codec),'InputImpl':SimpleNamespace(VideoFromFile=DownloadBoundary),'mimetype_to_extension':lambda mime:mime.split('/')[-1].lower()}
 api=ModuleType('pinned_wavespeed_api');sys.modules[api.__name__]=api;exec(compile((ROOT/FILES[1]).read_text(),str(ROOT/FILES[1]),'exec'),api.__dict__);namespace.update(api.__dict__)
 def uuid4():
  nonlocal uuid_number
  uuid_number+=1;return f'00000000-0000-4000-8000-{uuid_number:012d}'
 namespace['uuid']=SimpleNamespace(uuid4=uuid4)
 selected(FILES[6],{'common_upscale'},namespace)
 selected(FILES[3],{'bytesio_to_image_tensor','tensor_to_bytesio','tensor_to_pil','pil_to_bytesio','downscale_image_tensor','_compute_downscale_dims'},namespace)
 selected(FILES[5],{'get_number_of_images','validate_video_duration','validate_container_format_is_mp4'},namespace)
 async def upload(cls,stream,filename,mime,wait_label='Uploading',progress_origin_ts=None):
  index=len(uploads);uploads.append({'file_name':filename,'mime_type':mime,'wait_label':wait_label,'bytes_hex':stream.getvalue().hex()});events.append({'upload':index});return f'https://source.fixture.invalid/input-{index+1}.bin'
 namespace['upload_file_to_comfyapi']=upload
 selected(FILES[2],{'upload_images_to_comfyapi','upload_image_to_comfyapi','upload_video_to_comfyapi'},namespace)
 def normalize(value):
  if isinstance(value,Enum):return value.value
  if hasattr(value,'model_dump'):return value.model_dump(mode='json',exclude_none=False)
  if isinstance(value,torch.Tensor):return {'shape':list(value.shape),'dtype':str(value.dtype),'pixels':value.flatten().tolist()}
  if isinstance(value,DownloadBoundary):return {'bytes_hex':value.stream.getvalue().hex(),'owner':'VideoFromFile boundary; no PyAV decode'}
  if isinstance(value,(tuple,list)):return [normalize(item) for item in value]
  if isinstance(value,dict):return {key:normalize(item) for key,item in value.items()}
  return value
 async def sync(cls,endpoint,response_model,data=None,**kwargs):
  payload=copy.deepcopy(options.get('creation',{'code':200,'message':'created','data':{'id':'source-task-42'}}))
  record={'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(mode='json',exclude_none=True),'options':kwargs,'response':payload};requests.append(record);events.append({'request':endpoint.path})
  response=response_model.model_validate(payload);record['normalized_response']=response.model_dump(mode='json',exclude_none=True);return response
 async def poll(cls,endpoint,response_model,**kwargs):
  record={'path':endpoint.path,'method':endpoint.method,'options':{key:normalize(value) for key,value in kwargs.items() if not callable(value)},'responses':[],'statuses':[],'progress_updates':[],'delays':[]};polls.append(record)
  queue=copy.deepcopy(options.get('poll_responses',[{'code':200,'message':'working','data':{'status':'queued','outputs':[]}},{'code':200,'message':'working','data':{'status':'processing','outputs':[]}},{'code':200,'message':'done','data':{'status':'completed','outputs':['https://source.fixture.invalid/result.mp4' if options['video'] else 'https://source.fixture.invalid/result.png','https://source.fixture.invalid/unused.bin']}}]))
  if len(queue)==1:
   try:response_model.model_validate(queue[0])
   except Exception:queue=queue*480
  async def sync_poll(*args,**kw):
   if not queue:raise RuntimeError('oracle poll script exhausted')
   payload=queue.pop(0);record['responses'].append({'raw':copy.deepcopy(payload)});events.append({'poll_http':endpoint.path})
   try:response=response_model.model_validate(payload)
   except Exception:return payload
   record['responses'][-1]['normalized']=response.model_dump(mode='json',exclude_none=True);record['statuses'].append(kwargs['status_extractor'](response));return payload
  async def virtual_delay(seconds,*args,**kwargs):record['delays'].append(seconds)
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
 async def download(url,dest,**kwargs):
  data=VIDEO.read_bytes() if options['video'] else (b'not a source image' if options.get('malformed_image') else IMAGE.read_bytes())
  downloads.append({'url':url,'options':kwargs,'bytes_hex':data.hex()});events.append({'download':url});dest.write(data);dest.seek(0)
 namespace.update(sync_op=sync,poll_op=poll,download_url_to_bytesio=download,ApiEndpoint=lambda path,method='GET':SimpleNamespace(path=path,method=method))
 selected(FILES[4],{'download_url_to_video_output','download_url_to_image_tensor'},namespace)
 selected(FILES[0],set(NODES),namespace,True)
 for name in ['FlashVSRRequest','SeedVR2ImageRequest']:
  original=namespace[name]
  def constructor(*args,_name=name,_original=original,**kwargs):
   record={'model':_name,'raw':normalize(kwargs)};constructors.append(record);events.append({'constructor':len(constructors)-1});result=_original(*args,**kwargs);record['normalized']=result.model_dump(mode='json',exclude_none=True);return result
  namespace[name]=constructor
 class LowerBoundary:
  def __init__(self,value):self.value=value
  def lower(self):
   record={'raw':self.value};lowers.append(record);events.append({'lower':self.value});result=self.value.lower();record['resolved']=result;return result
 def base(node):return {'node':node,'settings':{'target_resolution':'2K',**({'model':'SeedVR2'} if node==NODES[1] else {})},'image':{'batch':1,'height':3,'width':2,'channels':3} if node==NODES[1] else None,'video':{'duration':8.,'container':'mp4'} if node==NODES[0] else None}
 def arguments(value):
  result=copy.deepcopy(value['settings']);result['target_resolution']=LowerBoundary(result['target_resolution'])
  if value['video'] is not None:result['video']=VideoBoundary(value['video'])
  else:
   descriptor=value['image'];image=torch.empty((descriptor['batch'],descriptor['height'],descriptor['width'],descriptor['channels']),dtype=torch.float32)
   for batch in range(descriptor['batch']):
    for channel in range(descriptor['channels']):image[batch,...,channel]=[.5,.25,.125,1.][channel]+(batch*.01 if channel<3 else 0.)
   result['image']=image
  return result
 cases=[];errors=[]
 async def observe(value):
  nonlocal uuid_number
  uuid_number=0
  for collection in [requests,constructors,polls,uploads,downloads,conversions,events,lowers]:collection.clear()
  options.clear();options.update(value.get('oracle_options',{}));options['video']=value['video'] is not None
  try:result=await namespace[value['node']].execute(**arguments(value))
  except (NameError,ImportError) as error:raise RuntimeError(f'harness failure {value["node"]}: {error}') from error
  except Exception as error:
   if 'virtual_delay' in str(error) or 'oracle poll script exhausted' in str(error):raise RuntimeError('oracle harness failure') from error
   errors.append({'input':copy.deepcopy(value),'error_type':type(error).__name__,'error_message':re.sub(r'<_io.BytesIO object at 0x[0-9a-f]+>', '<_io.BytesIO object at <address>>', str(error)),'requests':copy.deepcopy(requests),'constructors':copy.deepcopy(constructors),'polls':copy.deepcopy(polls),'uploads':copy.deepcopy(uploads),'downloads':copy.deepcopy(downloads),'conversions':copy.deepcopy(conversions),'events':copy.deepcopy(events),'lowers':copy.deepcopy(lowers)});return
  cases.append({'input':copy.deepcopy(value),'requests':copy.deepcopy(requests),'constructors':copy.deepcopy(constructors),'polls':copy.deepcopy(polls),'uploads':copy.deepcopy(uploads),'downloads':copy.deepcopy(downloads),'conversions':copy.deepcopy(conversions),'events':copy.deepcopy(events),'lowers':copy.deepcopy(lowers),'output':normalize(result)})
 for node in NODES:
  await observe(base(node))
  for resolution in ['720p','1080p','2K','4K','8K',' unknown ','İ','',None,123]:value=base(node);value['settings']['target_resolution']=resolution;await observe(value)
  for response in [{'code':500,'message':'source failure'},{'code':200,'message':'ok','data':None},{'code':200,'message':'ok','data':{}},{'code':'200','message':'ok','data':{'id':'source-task-42'}},{'code':200,'message':'ok','data':{'id':''}}]:value=base(node);value['oracle_options']={'creation':response};await observe(value)
  for response in [{'code':200,'message':'failed','data':None},{'code':200,'message':'failed','data':{'status':'failed'}},{'code':500,'message':'processing fail','data':{'status':'completed','outputs':['https://source.fixture.invalid/result.bin']}},{'code':200,'message':'empty','data':{'status':'completed'}},{'code':200,'message':'case','data':{'status':' SUCCEEDED ','outputs':['https://source.fixture.invalid/result.mp4' if node==NODES[0] else 'https://source.fixture.invalid/result.png']}},{'code':200,'message':'bad','data':{'status':123}},{'code':200,'message':'bad','data':{'status':'done','outputs':None}}]:value=base(node);value['oracle_options']={'poll_responses':[response]};await observe(value)
 for model in ['Ultimate','seedvr2','ultimate','unknown','',None,123]:value=base(NODES[1]);value['settings']['model']=model;await observe(value)
 for batch in [0,2,7]:value=base(NODES[1]);value['image']['batch']=batch;await observe(value)
 for channels in [1,4]:value=base(NODES[1]);value['image']['channels']=channels;await observe(value)
 for dimension in [0,1,32]:value=base(NODES[1]);value['image']['width']=dimension;await observe(value)
 value=base(NODES[1]);value['oracle_options']={'malformed_image':True};await observe(value)
 for container in ['mov,mp4,m4a,3gp,3g2,mj2','avi','MP4','']:value=base(NODES[0]);value['video']['container']=container;await observe(value)
 for duration in [4.9998,4.9999,5.,600.,600.0001,600.0002]:value=base(NODES[0]);value['video']['duration']=duration;await observe(value)
 for flag in ['duration_error','container_error','conversion_error']:value=base(NODES[0]);value['video'][flag]=True;await observe(value)
 schema={}
 for original in ast.parse((ROOT/FILES[0]).read_text()).body:
  if isinstance(original,ast.ClassDef) and original.name in NODES:schema[original.name]=[ast.unparse(member) for member in original.body if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)) and member.name=='define_schema']
 return {'sources':{file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in FILES},'asset_sha256':{'video':hashlib.sha256(VIDEO.read_bytes()).hexdigest(),'image':hashlib.sha256(IMAGE.read_bytes()).hexdigest()},'versions':{'torch':torch.__version__,'pillow':Image.__version__},'cases':cases,'errors':errors,'schema_facts':schema,'error_representation':'Only nondeterministic BytesIO address in Pillow UnidentifiedImageError is replaced by <address>; exception type/message and actual malformed bytes remain retained.', 'scope':'actual source nodes/API/PNGupload+RGBAimage download/clientpoll+extractor control and video helper boundary; retained real MP4 Video.save_to/VideoFromFile stand-in, virtual delay/UI; no PyAV/realclock/native allocation proof'}
parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();result=asyncio.run(generate());encoded=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
if args.check:
 if DEST.read_text()!=encoded:raise SystemExit('source oracle drift')
else:DEST.write_text(encoded)
print(f"Wavespeed actual source boundary: {len(result['cases'])} successes,{len(result['errors'])} ordered failures")
