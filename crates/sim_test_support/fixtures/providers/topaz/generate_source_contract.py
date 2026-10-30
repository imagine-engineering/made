"""Actual pinned Topaz AST/API/image helpers; native video metadata owner is separate."""
import argparse,ast,asyncio,builtins,copy,hashlib,inspect,json,math,os,sys,time
from io import BytesIO
from pathlib import Path
from types import ModuleType,SimpleNamespace
import aiohttp,numpy as np,torch
from PIL import Image
ROOT=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI');DEST=Path(__file__).with_name('source-contract.json')
FILES=['sim_api_nodes/nodes_topaz.py','sim_api_nodes/apis/topaz.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/upload_helpers.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/_helpers.py','sim_api_nodes/util/client.py','sim/utils.py','sim_api/latest/_input_impl/video_types.py']
NODES=['TopazImageEnhance','TopazVideoEnhance','TopazVideoEnhanceV2'];torch.set_num_threads(1)
VIDEO=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI-Frontend/browser_tests/assets/plain_video.mp4').read_bytes()
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
class Writer:
 def __init__(self):self.bytes=bytearray()
 async def write(self,bytes):self.bytes.extend(bytes)
async def generate():
 requests=[];polls=[];uploads=[];puts=[];downloads=[];events=[];options={}
 namespace={'__builtins__':__builtins__,'builtins':builtins,'torch':torch,'np':np,'Image':Image,'BytesIO':BytesIO,'math':math,'os':os,'time':time,'IO':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *values:values),'Input':SimpleNamespace(),'uuid':SimpleNamespace(uuid4=lambda:'00000000-0000-4000-8000-000000000000'),'mimetype_to_extension':lambda mime:mime.split('/')[-1].lower()}
 api=ModuleType('pinned_topaz_api');sys.modules[api.__name__]=api;exec(compile((ROOT/FILES[1]).read_text(),str(ROOT/FILES[1]),'exec'),api.__dict__);namespace.update(api.__dict__)
 for assignment in ast.parse((ROOT/FILES[0]).read_text()).body:
  if isinstance(assignment,ast.Assign):
   try:value=ast.literal_eval(assignment.value)
   except (ValueError,TypeError):continue
   for target in assignment.targets:
    if isinstance(target,ast.Name):namespace[target.id]=value
 selected(FILES[7],{'common_upscale'},namespace)
 selected(FILES[2],{'tensor_to_bytesio','tensor_to_pil','pil_to_bytesio','downscale_image_tensor','_compute_downscale_dims','bytesio_to_image_tensor'},namespace)
 selected(FILES[4],{'get_number_of_images','validate_container_format_is_mp4'},namespace)
 selected(FILES[5],{'get_fs_object_size'},namespace)
 async def upload_file(cls,stream,filename,mime,wait_label='Uploading',progress_origin_ts=None):
  uploads.append({'file_name':filename,'mime_type':mime,'bytes_hex':stream.getvalue().hex(),'wait_label':wait_label});events.append({'upload':len(uploads)-1});return f'https://source.fixture.invalid/input-{len(uploads)}.bin'
 namespace['upload_file_to_simapi']=upload_file;selected(FILES[3],{'upload_images_to_simapi'},namespace)
 # Execute the original source FormData branch, rather than a hand-written wire encoder.
 form_branch=next(node for node in ast.walk(ast.parse((ROOT/FILES[6]).read_text())) if isinstance(node,ast.If) and node.lineno==663)
 async def source_payload(body,method='POST',content_type='multipart/form-data'):
  form_namespace={'aiohttp':aiohttp,'cfg':SimpleNamespace(content_type=content_type,multipart_parser=None,data=body,files=None),'method':method,'payload_headers':{},'payload_kw':{}}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(form_branch)],type_ignores=[])),str(ROOT/FILES[6]),'exec'),form_namespace)
  if content_type!='multipart/form-data':return form_namespace['payload_kw'],form_namespace['payload_headers']
  form=form_namespace['payload_kw']['data'];fields=[{'name':attrs['name'],'value':value} for attrs,headers,value in form._fields]
  form._writer=aiohttp.MultipartWriter('form-data',boundary='sim-topaz-canonical-boundary');writer=Writer();await form().write(writer);return fields,writer.bytes.hex()
 async def sync(cls,endpoint,response_model,data=None,**kwargs):
  body=data.model_dump(mode='json',exclude_none=True) if data is not None else None;record={'path':endpoint.path,'method':endpoint.method,'body':body,'options':kwargs}
  if kwargs.get('content_type')=='multipart/form-data':record['form_fields'],record['wire_hex']=await source_payload(body)
  else:record['source_payload'],record['source_payload_headers']=await source_payload(body,endpoint.method,kwargs.get('content_type','application/json'))
  requests.append(record);events.append({'request':endpoint.path})
  if response_model is namespace['ImageAsyncTaskResponse']:payload={'process_id':'image-42'}
  elif response_model is namespace['ImageDownloadResponse']:payload={'download_url':'https://source.fixture.invalid/result-image.png','expiry':1790812800}
  elif response_model is namespace['CreateVideoResponse']:payload={'requestId':'video-42'}
  elif response_model is namespace['VideoAcceptResponse']:payload={'uploadId':'retained-upload-id','urls':['https://signed.fixture.invalid/upload-video.mp4']*options.get('signed_urls',1)}
  else:payload={'message':'upload complete'}
  if options.get('missing_response_id') and response_model in (namespace['ImageAsyncTaskResponse'],namespace['CreateVideoResponse']):payload={}
  record['response']=copy.deepcopy(payload)
  response=response_model.model_validate(payload);record['normalized_response']=response.model_dump(mode='json',exclude_none=True);return response
 async def poll(cls,endpoint=None,response_model=None,**kwargs):
  endpoint=endpoint if endpoint is not None else kwargs.pop('poll_endpoint')
  record={'path':endpoint.path,'method':endpoint.method,'poll_interval':kwargs.get('poll_interval'),'estimated_duration':kwargs.get('estimated_duration'),'statuses':[],'responses':[]};polls.append(record);events.append({'poll':endpoint.path})
  for status,progress in [('queued',0),('RUNNING',25),('SUCCEEDED',100)]:
   if response_model is namespace['ImageStatusResponse']:payload={'process_id':'image-42','status':status,'progress':progress,'credits':options.get('credits',3)}
   else:payload={'status':status,'progress':progress,'estimates':None if options.get('no_estimate') else {'cost':options.get('cost',[7,99])},'message':'source-status','download':None if options.get('no_download') else {'url':'https://source.fixture.invalid/result-video.mp4'}}
   response=response_model.model_validate(payload);record['responses'].append({'raw':copy.deepcopy(payload),'normalized':response.model_dump(mode='json',exclude_none=True)});record['statuses'].append({'status':kwargs['status_extractor'](response),'progress':kwargs['progress_extractor'](response),'price':kwargs['price_extractor'](response)})
  return response
 class VideoOwnerProbe:
  def __init__(self,value):self.value=value;self.stream=BytesIO(VIDEO);self.stream.seek(3)
  def read(self,key):
   events.append({'metadata':key})
   if self.value.get('failure')==key:raise RuntimeError(f'source native {key} owner unavailable')
   return self.value[key]
  def get_container_format(self):return self.read('container')
  def get_dimensions(self):return self.read('width'),self.read('height')
  def get_frame_rate(self):return self.read('fps')
  def get_duration(self):return self.read('duration')
  def get_frame_count(self):return self.read('frames')
  def get_stream_source(self):
   if self.value.get('stream_kind')=='file':events.append({'stream_source':'file'});return str(ROOT.parent/'ComfyUI-Frontend/browser_tests/assets/plain_video.mp4')
   events.append({'stream_source':'BytesIO'});return self.stream
 class SignedResponse:
  def __init__(self):self.headers={} if options.get('missing_etag') else {'Etag':'"source-etag-42"'}
  async def __aenter__(self):
   puts[-1]['response_headers']=copy.deepcopy(self.headers)
   if options.get('put_failure'):puts[-1]['error']='source signed PUT failed';raise RuntimeError('source signed PUT failed')
   return self
  async def __aexit__(self,*args):return False
 class Session:
  def __init__(self,headers):self.headers=headers
  async def __aenter__(self):return self
  async def __aexit__(self,*args):return False
  def put(self,url,data,raise_for_status):
   puts.append({'url':url,'headers':self.headers,'bytes_hex':data.read().hex(),'raise_for_status':raise_for_status});events.append({'put':url});return SignedResponse()
 namespace['aiohttp']=SimpleNamespace(ClientSession=Session)
 image_stream=BytesIO();Image.new('RGBA',(2,2),(128,64,32,128)).save(image_stream,format='PNG');PNG=image_stream.getvalue()
 async def download_image(url):downloads.append({'url':url,'kind':'image','bytes_hex':PNG.hex()});events.append({'download':url});return namespace['bytesio_to_image_tensor'](BytesIO(PNG))
 async def download_video(url):downloads.append({'url':url,'kind':'video','bytes_hex':VIDEO.hex()});events.append({'download':url});return {'video_owner_url':url}
 namespace.update(sync_op=sync,poll_op=poll,download_url_to_image_tensor=download_image,download_url_to_video_output=download_video,ApiEndpoint=lambda path,method='GET':SimpleNamespace(path=path,method=method))
 selected(FILES[0],set(NODES),namespace,True)
 def base(node):
  return {'node':node,'settings':{} if node=='TopazImageEnhance' else ({'upscaler_enabled':True,'upscaler_model':'Starlight (Astra) Fast','upscaler_resolution':'FullHD (1080p)'} if node=='TopazVideoEnhance' else {'upscaler_model':{'upscaler_model':'Astra 2','upscaler_resolution':'FullHD (1080p)','creativity':.5,'prompt':'','sharp':.5,'realism':0.},'interpolation_model':{'interpolation_model':'Disabled'}}),'image':{'batch':1,'width':2,'height':2,'channels':3} if node=='TopazImageEnhance' else None,'video':None if node=='TopazImageEnhance' else {'container':'mp4','width':1920,'height':1080,'fps':23.976,'duration':5.99,'frames':144}}
 def arguments(value):
  result=copy.deepcopy(value['settings'])
  if value['image'] is not None:
   descriptor=value['image'];tensor=torch.empty((descriptor['batch'],descriptor['height'],descriptor['width'],descriptor['channels']),dtype=torch.float32)
   for channel in range(descriptor['channels']):tensor[...,channel]=[.5,.25,.125,1.][channel]
   result.setdefault('model','Reimagine');result['image']=tensor
  else:result['video']=VideoOwnerProbe(value['video'])
  return result
 def normalize(value):
  if isinstance(value,torch.Tensor):return {'shape':list(value.shape),'pixels':value.flatten().tolist()}
  if isinstance(value,(tuple,list)):return [normalize(item) for item in value]
  return value
 cases=[];errors=[]
 async def observe(value):
  for output in [requests,polls,uploads,puts,downloads,events]:output.clear()
  options.clear();options.update(value.get('oracle_options',{}))
  try:result=await namespace[value['node']].execute(**arguments(value))
  except (NameError,ImportError,TypeError) as error:raise RuntimeError(f'oracle harness failure: {value["node"]}: {error}') from error
  except Exception as error:errors.append({'input':copy.deepcopy(value),'error_type':type(error).__name__,'error_message':str(error),'requests':copy.deepcopy(requests),'polls':copy.deepcopy(polls),'uploads':copy.deepcopy(uploads),'puts':copy.deepcopy(puts),'downloads':copy.deepcopy(downloads),'events':copy.deepcopy(events)});return
  cases.append({'input':copy.deepcopy(value),'requests':copy.deepcopy(requests),'polls':copy.deepcopy(polls),'uploads':copy.deepcopy(uploads),'puts':copy.deepcopy(puts),'downloads':copy.deepcopy(downloads),'events':copy.deepcopy(events),'output':normalize(result)})
 for node in NODES:await observe(base(node))
 for batch in [0,2]:value=base(NODES[0]);value['image']['batch']=batch;await observe(value)
 for settings in [{'subject_detection':subject} for subject in ['All','Foreground','Background','arbitrary']]+[{'prompt':''},{'prompt':None},{'prompt':' ☁ '},{'face_enhancement':False},{'face_preservation':False,'color_preservation':False},{'face_enhancement_creativity':0.,'face_enhancement_strength':1.},{'output_width':32000},{'output_width':-1,'output_height':123},{'creativity':10},{'crop_to_fill':True}]:value=base(NODES[0]);value['settings']=settings;await observe(value)
 for number in [0.000001,1e16,-0.0,-.1,1.5]:value=base(NODES[0]);value['settings']['face_enhancement_creativity']=number;await observe(value)
 for option in [{'credits':0},{'credits':-1},{'missing_response_id':True}]:value=base(NODES[0]);value['oracle_options']=option;await observe(value)
 for node in NODES[1:]:value=base(node);value['video']['stream_kind']='file';await observe(value)
 for model in namespace['UPSCALER_MODELS_MAP']:
  for resolution in ['FullHD (1080p)','4K (2160p)','source fallback']:
   for interpolation in [False,True]:
    for node in NODES[1:]:
     value=base(node)
     if node==NODES[1]:value['settings'].update(upscaler_model=model,upscaler_resolution=resolution,interpolation_enabled=interpolation)
     else:value['settings']['upscaler_model'].update(upscaler_model=model,upscaler_resolution=resolution,creativity='middle' if model=='Starlight (Astra) Creative' else .5);value['settings']['interpolation_model']={'interpolation_model':'apo-8','interpolation_frame_rate':60,'interpolation_slowmo':2,'interpolation_duplicate':True,'interpolation_duplicate_threshold':.01} if interpolation else {'interpolation_model':'Disabled'}
     await observe(value)
 for width,height in [(1080,1920),(1000,1000),(321,123),(123,321),(4000,500),(500,4000)]:
  for node in NODES[1:]:value=base(node);value['video'].update(width=width,height=height);await observe(value)
 for node in NODES[1:]:
  value=base(node)
  if node==NODES[1]:value['settings'].update(upscaler_enabled=False,interpolation_enabled=False)
  else:value['settings'].update(upscaler_model={'upscaler_model':'Disabled'},interpolation_model={'interpolation_model':'Disabled'})
  await observe(value)
  for container in ['mov,mp4,m4a,3gp,3g2,mj2','matroska']:value=base(node);value['video']['container']=container;await observe(value)
  for key in ['container','width','fps','duration','frames']:value=base(node);value['video']['failure']=key;await observe(value)
  for option in [{'signed_urls':0},{'signed_urls':2},{'missing_etag':True},{'put_failure':True},{'missing_response_id':True},{'cost':[]},{'cost':[0]},{'no_estimate':True},{'no_download':True}]:value=base(node);value['oracle_options']=option;await observe(value)
 for frames in [449,450,451,9000,9001]:
  for prompt in ['','  ',' descriptive scene ']:value=base(NODES[2]);value['video']['frames']=frames;value['settings']['upscaler_model']['prompt']=prompt;await observe(value)
 schemas={node.name:ast.unparse(next(member for member in node.body if isinstance(member,ast.FunctionDef) and member.name=='define_schema')) for node in ast.parse((ROOT/FILES[0]).read_text()).body if isinstance(node,ast.ClassDef) and node.name in NODES}
 return {'sources':{file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in FILES},'versions':{'torch':torch.__version__,'pillow':Image.__version__,'aiohttp':aiohttp.__version__},'cases':cases,'errors':errors,'schema_facts':schemas,'encoded_video_sha256':hashlib.sha256(VIDEO).hexdigest(),'scope':'actual source node/API/image conversion/upload/multipart; structural native video metadata/stream probe; native PyAV/video decode and allocation unclaimed'}
parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();result=asyncio.run(generate());encoded=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
if args.check:
 if DEST.read_text()!=encoded:raise SystemExit('source oracle drift')
else:DEST.write_text(encoded)
print(f"Topaz actual source boundary: {len(result['cases'])} successes,{len(result['errors'])} ordered failures; native PyAV separate")
