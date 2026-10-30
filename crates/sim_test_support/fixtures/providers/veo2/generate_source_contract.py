import argparse,ast,asyncio,base64,contextlib,copy,hashlib,io,json,logging,math,re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
import aiohttp
import numpy as np
from PIL import Image
SOURCE=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI');ROOT=Path(__file__).resolve().parent
FRONTEND=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI-Frontend');MP4=(FRONTEND/'browser_tests/assets/plain_video.mp4').read_bytes()
PATHS=['sim_api_nodes/nodes_veo2.py','sim_api_nodes/apis/veo.py','sim_api_nodes/util/client.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/download_helpers.py','sim_api_nodes/util/_helpers.py','sim_api/latest/_input_impl/video_types.py']
class Tensor:
 def __init__(self,slot='image',count=1,channels=3,array=None):self.slot=slot;self.array=np.tile(np.array([.5,.25,.125]+([.375] if channels==4 else []),dtype=np.float32),(count,2,2,1)) if array is None else array
 @property
 def shape(self):return self.array.shape
 def __getitem__(self,index):return Tensor(self.slot+':'+str(index),array=self.array[index])
 def cpu(self):return self
 def numpy(self):return self.array
 def unsqueeze(self,dimension):return Tensor(self.slot,array=np.expand_dims(self.array,dimension))
 def squeeze(self):return Tensor(self.slot,array=np.squeeze(self.array))
 def movedim(self,origin,target):return Tensor(self.slot,array=np.moveaxis(self.array,origin,target))
def serial(value):
 if isinstance(value,Tensor):return {'native_slot':value.slot,'shape':list(value.shape),'pixels':value.array.reshape(-1,value.shape[-1])[0].tolist() if value.array.size else []}
 if isinstance(value,dict):return {key:serial(member) for key,member in value.items()}
 if isinstance(value,(tuple,list)):return [serial(member) for member in value]
 return value
def strip(node):
 for member in ast.walk(node):
  if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
   member.returns=None
   for argument in member.args.args+member.args.kwonlyargs:argument.annotation=None
 return ast.fix_missing_locations(node)
async def generate():
 logging.disable(logging.CRITICAL);requests=[];media=[];downloads=[];delays=[];progress=[];polls=[];overrides={};contracts={}
 namespace={'__builtins__':__builtins__,'__name__':'actual_veo_source_probe','base64':base64,'BytesIO':io.BytesIO,'Image':Image,'np':np,'math':math,'json':json,'asyncio':asyncio,'contextlib':contextlib,'logging':logging,'dataclass':dataclass,'IO':SimpleNamespace(ComfyNode=type('ComfyNode',(),{}),NodeOutput=lambda *values:values),'time':SimpleNamespace(monotonic=lambda:0.0),'is_processing_interrupted':lambda:False}
 hidden=SimpleNamespace(auth_token_sim_org='capability-token',api_key_sim_org='capability-api-key',sim_usage_source=None)
 namespace['IO'].ComfyNode.hidden=hidden;namespace['get_deploy_environment']=lambda:'host'
 helpers=ast.parse((SOURCE/PATHS[5]).read_text());selected=[copy.deepcopy(node) for node in helpers.body if isinstance(node,ast.FunctionDef) and node.name in ['get_auth_header','get_usage_source','get_sim_api_headers']];exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<actual-Veo-auth-headers>','exec'),namespace)
 auth_header_probes=[]
 for token,api_key in [('capability-token','capability-api-key'),(None,'capability-api-key'),(None,None)]:
  hidden.auth_token_sim_org=token;hidden.api_key_sim_org=api_key;auth_header_probes.append({'auth_token_available':bool(token),'api_key_available':bool(api_key),'headers':namespace['get_sim_api_headers'](namespace['IO'].ComfyNode)})
 hidden.auth_token_sim_org='capability-token';hidden.api_key_sim_org='capability-api-key'
 exec(compile((SOURCE/PATHS[1]).read_text(),'<actual-Veo-api-models>','exec'),namespace)
 actual_request_model=namespace['VeoGenVidRequest'];construction_probes=[]
 def model_input(value):
  if hasattr(value,'model_dump'):return value.model_dump(mode='json',exclude_none=True)
  if isinstance(value,dict):return {key:model_input(member) for key,member in value.items()}
  if isinstance(value,(tuple,list)):return [model_input(member) for member in value]
  return value
 def request_constructor(**kwargs):
  probe={'kind':'Request','input':model_input(kwargs)}
  try:
   result=actual_request_model(**kwargs);probe['output']=result.model_dump(mode='json',exclude_none=True)
  except Exception as error:
   probe['error_type']=type(error).__name__
   if probe not in construction_probes:construction_probes.append(probe)
   raise
  if probe not in construction_probes:construction_probes.append(probe)
  return result
 namespace['VeoGenVidRequest']=request_constructor
 video_tree=ast.parse((SOURCE/PATHS[6]).read_text());video_class=next(node for node in video_tree.body if isinstance(node,ast.ClassDef) and node.name=='VideoFromFile');video=ast.ClassDef(name='VideoFromFile',bases=[],keywords=[],decorator_list=[],body=[copy.deepcopy(member) for member in video_class.body if isinstance(member,ast.FunctionDef) and member.name in ['__init__','get_stream_source']]);namespace['io']=io;exec(compile(strip(ast.Module(body=[video],type_ignores=[])),'<actual-VideoFromFile-byte-wrapper>','exec'),namespace);namespace['InputImpl']=SimpleNamespace(VideoFromFile=namespace['VideoFromFile'])
 conversions=ast.parse((SOURCE/PATHS[3]).read_text());members=[copy.deepcopy(node) for node in conversions.body if isinstance(node,ast.FunctionDef) and node.name in ['tensor_to_pil','tensor_to_base64_string','pil_to_bytesio','_compute_downscale_dims','downscale_image_tensor']];exec(compile(strip(ast.Module(body=members,type_ignores=[])),'<actual-tensor-PIL-base64-no-Torch-allocation>','exec'),namespace);actual_image=namespace['tensor_to_base64_string']
 def image_owner(tensor):
  record={'image':serial(tensor),'total_pixels':4194304,'mime_type':'image/png'};media.append(record)
  encoded=actual_image(tensor);record.update(base64=encoded,bytes_hex=base64.b64decode(encoded).hex());return encoded
 namespace['tensor_to_base64_string']=image_owner
 client=ast.parse((SOURCE/PATHS[2]).read_text());selected=[]
 for node in client.body:
  if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in ['COMPLETED_STATUSES','FAILED_STATUSES','QUEUED_STATUSES'] for target in node.targets):selected.append(copy.deepcopy(node))
  elif isinstance(node,ast.ClassDef) and node.name=='_PollUIState':selected.append(copy.deepcopy(node))
  elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ['sync_op','poll_op','poll_op_raw','_normalize_statuses','_normalize_status_value','_validate_or_raise','_wrap_model_extractor']:selected.append(copy.deepcopy(node))
 exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<actual-Veo-sync-poll-wrappers>','exec'),namespace)
 class Interrupted(Exception):pass
 class NetworkError(Exception):pass
 class ApiError(Exception):pass
 async def delay(seconds,*args):delays.append(seconds)
 namespace.update(ProcessingInterrupted=Interrupted,LocalNetworkError=NetworkError,ApiServerError=ApiError,sleep_with_interrupt=delay,_display_time_progress=lambda *args,**kwargs:progress.append(kwargs),utils=SimpleNamespace(ProgressBar=lambda total:SimpleNamespace(update_absolute=lambda *args,**kwargs:None)),ApiEndpoint=lambda path,method='GET',**kwargs:SimpleNamespace(path=path,method=method,headers={}))
 initial={'name':'projects/source/operations/task-1'};final={'done':True,'response':{'raiMediaFilteredCount':0,'videos':[{'bytesBase64Encoded':base64.b64encode(MP4).decode(),'gcsUri':'https://fixture.invalid/video.mp4','mimeType':'video/mp4'},{'gcsUri':'https://fixture.invalid/ignored.mp4'}]}}
 async def raw_sync(cls,endpoint,data=None,**options):
  body=data.model_dump(mode='json',exclude_none=True) if data else None;record={'header_order':list(namespace['get_sim_api_headers'](cls))+['Content-Type'],'headers':{**namespace['get_sim_api_headers'](cls),'Content-Type':'application/json'},'path':endpoint.path,'method':endpoint.method,'body':body,'json_body_hex':aiohttp.payload.JsonPayload(body)._value.hex(),'options':options};requests.append(record)
  if endpoint.path.endswith('/generate'):return copy.deepcopy(overrides.get('generate',initial))
  sequence=overrides.get('polls',[final]);index=min(len(polls),len(sequence)-1);value=copy.deepcopy(sequence[index]);polls.append(value);return value
 namespace['sync_op_raw']=raw_sync
 async def download_bytes(url,destination,**options):
  downloads.append({'url':url,'options':options})
  parsed=urlparse(url)
  if not parsed.scheme and not parsed.netloc and options.get('cls') is None:raise ValueError("For relative 'cloud' paths, the `cls` parameter is required.")
  destination.write(MP4);destination.seek(0)
 namespace['download_url_to_bytesio']=download_bytes
 download_tree=ast.parse((SOURCE/PATHS[4]).read_text());method=next(node for node in download_tree.body if isinstance(node,ast.AsyncFunctionDef) and node.name=='download_url_to_video_output');exec(compile(strip(ast.Module(body=[copy.deepcopy(method)],type_ignores=[])),'<actual-Veo-download-video-wrapper>','exec'),namespace)
 tree=ast.parse((SOURCE/PATHS[0]).read_text());nodes=['VeoVideoGenerationNode','Veo3VideoGenerationNode','Veo3FirstLastFrameNode'];selected=[]
 for node in tree.body:
  if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in ['MODELS_MAP','AVERAGE_DURATION_VIDEO_GEN'] for target in node.targets):selected.append(copy.deepcopy(node))
  elif isinstance(node,ast.ClassDef) and node.name in nodes:
   method=next(copy.deepcopy(member) for member in node.body if isinstance(member,ast.AsyncFunctionDef) and member.name=='execute');schema=next(call for call in ast.walk(node) if isinstance(call,ast.Call) and ast.unparse(call.func)=='IO.Schema');fields={keyword.arg:keyword.value for keyword in schema.keywords};defaults={argument.arg:ast.literal_eval(default) for argument,default in zip(method.args.args[-len(method.args.defaults):],method.args.defaults)} if method.args.defaults else {};contracts[node.name]={'arguments':[argument.arg for argument in method.args.args if argument.arg!='cls'],'defaults':defaults,'schema':ast.unparse(schema),'hidden':ast.unparse(fields['hidden']),'outputs':ast.unparse(fields['outputs']),'price_badge':ast.unparse(fields['price_badge'])};selected.append(ast.ClassDef(name=node.name,bases=[ast.Attribute(value=ast.Name(id='IO',ctx=ast.Load()),attr='ComfyNode',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[]))
 exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<actual-Veo-registered-execute>','exec'),namespace)
 bases={'VeoVideoGenerationNode':{'prompt':' source prompt '},'Veo3VideoGenerationNode':{'prompt':' source prompt '},'Veo3FirstLastFrameNode':{'prompt':' source prompt ','negative_prompt':'','resolution':'720p','aspect_ratio':'16:9','duration':8,'seed':0,'first_frame':Tensor('first_frame',2),'last_frame':Tensor('last_frame',1,4),'model':'veo-3.1-generate','generate_audio':True}}
 cases=[];errors=[];response_errors=[]
 async def observe(node,arguments,failure=False,override=None):
  requests.clear();media.clear();downloads.clear();delays.clear();progress.clear();polls.clear();overrides.clear();overrides.update(override or {})
  record={'input':{'node':node,'arguments':serial(arguments)},'override':override or {}}
  try:output=await namespace[node].execute(**copy.deepcopy(arguments))
  except Exception as error:
   if not failure:raise
   record.update(error_type=type(error).__name__,message=re.sub(r'0x[0-9a-fA-F]+','<address>',str(error)));(response_errors if override else errors).append(record)
  else:
   if failure:raise ValueError('Expected source failure '+node)
   data=output[0].get_stream_source().getvalue();record['outputs']=[{'encoded_video_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'source_wrapper':'actual VideoFromFile __init__/get_stream_source; no PyAV parse/native allocation'}];cases.append(record)
  record.update(requests=copy.deepcopy(requests),media=copy.deepcopy(media),downloads=copy.deepcopy(downloads),delays=copy.deepcopy(delays),progress=copy.deepcopy(progress),polls=copy.deepcopy(polls))
 for node,base in bases.items():
  await observe(node,base)
  for model in namespace['MODELS_MAP']:
   for resolution in ['720p','1080p','4k'] if node!='VeoVideoGenerationNode' else ['unused']:
    options={**base,'model':model}
    if node!='VeoVideoGenerationNode':options['resolution']=resolution
    denied=resolution=='4k' and ('lite' in model or node=='Veo3VideoGenerationNode' and '3.0' in model)
    await observe(node,options,denied)
  if node!='Veo3FirstLastFrameNode':
   for image in [Tensor('image',1),Tensor('image',2,4),Tensor('image',0)]:await observe(node,{**base,'image':image},image.shape[0]==0)
   for seed in [-1,0,1,4294967295,'bad']:await observe(node,{**base,'seed':seed,'negative_prompt':' negative ','enhance_prompt':False,'generate_audio':True},seed=='bad')
   for prompt in ['',None,123,'\u001c source \u001f']:await observe(node,{**base,'prompt':prompt,'image':Tensor('image')},prompt is None or prompt==123)
   for duration in [0,-1,361,'8','bad',None]:await observe(node,{**base,'duration_seconds':duration},duration=='bad')
  else:
   await observe(node,{**base,'seed':-1,'negative_prompt':'','duration':0,'generate_audio':False})
   for field in ['first_frame','last_frame']:await observe(node,{**base,field:Tensor(field,0)},True)
   for field,value in [('prompt',None),('duration','bad')]:await observe(node,{**base,field:value},True)
  for model in ['unknown','unknown-3.0','unknown-lite']:await observe(node,{**base,'model':model,**({'resolution':'4k'} if node!='VeoVideoGenerationNode' else {})},True)
  for override in [{'generate':{}},{'generate':{'name':None}},{'polls':[{'done':False},{'done':True,**{key:value for key,value in final.items() if key!='done'}}]},{'polls':[{'done':False}]},{'polls':[{'done':'invalid'}]}]:await observe(node,base,'generate' in override or override['polls'][-1].get('done')!=True,override)
  for response in [None,{}, {'raiMediaFilteredCount':0,'videos':[]},{'raiMediaFilteredCount':0,'videos':[{}]},{'raiMediaFilteredCount':1,'raiMediaFilteredReasons':['source reason'],'videos':final['response']['videos']},{'raiMediaFilteredCount':2,'videos':final['response']['videos']},{'raiMediaFilteredCount':-1,'videos':final['response']['videos']},{'videos':final['response']['videos']},{'raiMediaFilteredCount':0,'videos':[{'bytesBase64Encoded':'a'}]},{'raiMediaFilteredCount':0,'videos':[{'gcsUri':'/cloud/video.mp4'}]},{'raiMediaFilteredCount':0,'videos':[{'gcsUri':'https://fixture.invalid/video.mp4'}]},{'raiMediaFilteredCount':0,'videos':[{'bytesBase64Encoded':base64.b64encode(MP4).decode()+'$! \t','gcsUri':'https://fixture.invalid/ignored.mp4'}]}]:
   success=bool(response is not None and (response.get('raiMediaFilteredCount') in [0,-1] if node=='VeoVideoGenerationNode' else response.get('raiMediaFilteredCount') in [0,None]) and response.get('videos') and (response['videos'][0].get('gcsUri','').startswith('https://') or response['videos'][0].get('bytesBase64Encoded') not in [None,'','a']))
   await observe(node,base,not success,{'polls':[{'done':True,'response':response}]})
  for error in [{},{'code':429,'message':'source failure'}]:await observe(node,base,True,{'polls':[{'done':True,'error':error,'response':final['response']}]})
 api_probes=list(construction_probes);base64_probes=[];scalar_probes=[]
 for record in cases+errors+response_errors:
  for request in record['requests']:
   if request['path'].endswith('/generate'):
    probe={'kind':'Request','input':request['body'],'output':actual_request_model.model_validate(request['body']).model_dump(mode='json',exclude_none=True)}
    if probe not in api_probes:api_probes.append(probe)
  for kind,model,value in [('GenerateResponse','VeoGenVidResponse',record['override'].get('generate',initial))]+[('PollResponse','VeoGenVidPollResponse',value) for value in record['polls']]:
   probe={'kind':kind,'input':value}
   try:probe['output']=namespace[model].model_validate(value).model_dump(mode='json',by_alias=True)
   except Exception as error:probe['error_type']=type(error).__name__
   if probe not in api_probes:api_probes.append(probe)
   if kind=='PollResponse' and 'output' in probe:
    output=probe['output'];response=output.get('response') or {};error=output.get('error') or {}
    for value in [response.get('raiMediaFilteredCount'),error.get('message'),error.get('code')]+(response.get('raiMediaFilteredReasons') or []):
     scalar={'value':value,'output':str(value)}
     if scalar not in scalar_probes:scalar_probes.append(scalar)
    for video in response.get('videos') or []:
     if video.get('bytesBase64Encoded'):
      encoded={'value':video['bytesBase64Encoded']}
      try:encoded['bytes_hex']=base64.b64decode(video['bytesBase64Encoded']).hex()
      except Exception as error:encoded['error_type']=type(error).__name__
      if encoded not in base64_probes:base64_probes.append(encoded)
 # Validation probes use actual models on raw constructed requests, including failing node arguments.
 for node,base in bases.items():
  for prompt in [' source prompt ',None,123]:
   raw={'instances':[{'prompt':prompt}],'parameters':{'durationSeconds':'8','seed':0,'generateAudio':'false','enhancePrompt':False,'aspectRatio':'custom'}};probe={'kind':'Request','input':raw}
   try:probe['output']=actual_request_model.model_validate(raw).model_dump(mode='json',exclude_none=True)
   except Exception as error:probe['error_type']=type(error).__name__
   if probe not in api_probes:api_probes.append(probe)
 return {'schema_version':1,'namespace':'zed.sim.provider.veo2','scope':'actual execute/model/generic-poll/PIL/base64/VideoFromFile wrapper with aiohttp3.11.8 JSON wire; no native codec allocation or actual clock','aiohttp_version':aiohttp.__version__,'source_sha256':{path:hashlib.sha256((SOURCE/path).read_bytes()).hexdigest() for path in PATHS},'source_history':'checksum-locked source snapshot; no ComfyUI Git metadata or original per-file history available','auth_header_probes':auth_header_probes,'models':namespace['MODELS_MAP'],'contracts':contracts,'cases':cases,'preflight_errors':errors,'response_errors':response_errors,'api_probes':api_probes,'base64_probes':base64_probes,'scalar_probes':scalar_probes,'encoded_video':{'source':'ComfyUI-Frontend/browser_tests/assets/plain_video.mp4','bytes':len(MP4),'sha256':hashlib.sha256(MP4).hexdigest(),'classification':'actual source BytesIO wrapper only; no PyAV parse/native VIDEO success'},'generation_response':initial,'poll_response':final,'python_helper_defaults':{name:{key.arg:ast.unparse(default) for key,default in zip(member.args.kwonlyargs,member.args.kw_defaults) if default is not None} for name in ['sync_op','poll_op'] for member in client.body if isinstance(member,ast.AsyncFunctionDef) and member.name==name}}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();data=asyncio.run(generate());encoded=json.dumps(data,indent=2,ensure_ascii=False)+'\n';target=ROOT/'source-contract.json'
 if args.check:
  if target.read_text()!=encoded:raise ValueError('Veo2 actual source changed')
 else:target.write_text(encoded)
 print('Veo2',len(data['cases']),'success',len(data['preflight_errors']),'preflight',len(data['response_errors']),'response failures')
