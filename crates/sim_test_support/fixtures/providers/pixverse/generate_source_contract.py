import argparse,ast,asyncio,contextlib,copy,hashlib,io,itertools,json,re,types,sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
FOLDER=Path(__file__).resolve().parent
SOURCE_ROOT=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI')
SOURCES=['sim_api_nodes/nodes_pixverse.py','sim_api_nodes/apis/pixverse.py','sim_api_nodes/util/client.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/download_helpers.py','sim_api_nodes/util/_helpers.py']
class TensorProbe:
 def __init__(self,slot='image',count=1):self.slot=slot;self.shape=(count,720,720,3)
def serial(value):
 if isinstance(value,TensorProbe):return {'native_slot':value.slot,'shape':list(value.shape)}
 if isinstance(value,dict):return {key:serial(member) for key,member in value.items()}
 if isinstance(value,(list,tuple)):return [serial(member) for member in value]
 return value
def strip(node):
 for member in ast.walk(node):
  if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
   member.returns=None
   for argument in member.args.args+member.args.kwonlyargs:argument.annotation=None
 return ast.fix_missing_locations(node)
async def generate():
 module=types.ModuleType('pixverse_pinned_api_oracle');sys.modules[module.__name__]=module;namespace=module.__dict__
 exec(compile((SOURCE_ROOT/SOURCES[1]).read_text(),'<actual-source-api>','exec'),namespace)
 for value in list(namespace.values()):
  if isinstance(value,type) and value.__module__==module.__name__ and hasattr(value,'model_rebuild'):value.model_rebuild(_types_namespace=namespace)
 requests=[];media=[];polls=[];downloads=[];override={};contracts={};png=(FOLDER/'downloaded-image.png').read_bytes()
 namespace.update(IO=SimpleNamespace(NodeOutput=lambda *values:values),torch=SimpleNamespace(Tensor=TensorProbe),ApiEndpoint=lambda path,method='GET':SimpleNamespace(path=path,method=method))
 tree=ast.parse((SOURCE_ROOT/SOURCES[0]).read_text());client=ast.parse((SOURCE_ROOT/SOURCES[2]).read_text())
 branch=next(copy.deepcopy(node) for node in ast.walk(client) if isinstance(node,ast.If) and ast.unparse(node.test)=="cfg.content_type == 'multipart/form-data' and method != 'GET'");branch.orelse=[];wire=compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),'<actual-source-multipart>','exec')
 class FormProbe:
  def __init__(self,default_to_multipart):self.fields=[];assert default_to_multipart
  def add_field(self,name,value,filename=None,content_type=None):self.fields.append({'name':name,'file_name':filename,'media_type':content_type,'bytes':len(value.getvalue()),'sha256':hashlib.sha256(value.getvalue()).hexdigest()})
 def prepare(image):
  media.append({'operation':'png_prepare','image':serial(image),'mime_type':'image/png','total_pixels':4194304,'first_batch':True})
  if image.shape[0]==0:raise IndexError('source first batch image is absent')
  value=io.BytesIO(png);value.name='00000000-0000-4000-8000-000000000000.png';return value
 async def sync(cls,endpoint,response_model,data=None,**options):
  record={'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(mode='json',exclude_none=True) if data is not None else None};requests.append(record)
  if 'files' in options:
   state={'cfg':SimpleNamespace(content_type=options['content_type'],multipart_parser=None,data=None,files=options['files']),'method':endpoint.method,'payload_headers':{},'payload_kw':{},'aiohttp':SimpleNamespace(FormData=FormProbe),'BytesIO':io.BytesIO,'contextlib':contextlib};exec(wire,state);record['wire_fields']=state['payload_kw']['data'].fields
   payload={'ErrCode':0,'ErrMsg':'source','Resp':{'img_id':100+sum(row['path'].endswith('/upload') for row in requests)}};payload.update(override.get('upload',{}))
  else:payload={'ErrCode':0,'ErrMsg':'source','Resp':{'video_id':42}};payload.update(override.get('generation',{}))
  return response_model.model_validate(payload)
 async def poll(cls,endpoint,response_model,**options):
  payload={'ErrCode':0,'ErrMsg':'source','Resp':{'status':1,'url':'https://media.fixture.invalid/video.mp4'}};payload.update(override.get('poll',{}));response=response_model.model_validate(payload)
  try:status=options['status_extractor'](response)
  except Exception:status=None
  polls.append({'path':endpoint.path,'method':endpoint.method,'estimated_duration':options['estimated_duration'],'completed_statuses':serial(options['completed_statuses']),'failed_statuses':serial(options['failed_statuses']),'status':status})
  if status in options['failed_statuses']:raise Exception('Task failed: source numeric status '+str(status))
  if status!=1:raise Exception('Source poll continuation gate: '+str(status))
  return response
 async def download(url):
  downloads.append({'url':url,'timeout':None,'max_retries':5,'source_wrapper':'InputImpl.VideoFromFile'})
  if not isinstance(url,str) or not url:raise ValueError('canonical download URL rejection')
  return {'canonical_video_from_file':url}
 namespace.update(sync_op=sync,poll_op=poll,tensor_to_bytesio=prepare,download_url_to_video_output=download)
 validation=ast.parse((SOURCE_ROOT/SOURCES[3]).read_text());exec(compile(strip(ast.Module(body=[copy.deepcopy(node) for node in validation.body if isinstance(node,ast.FunctionDef) and node.name=='validate_string'],type_ignores=[])),'<actual-source-validation>','exec'),namespace)
 extension=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='PixVerseExtension');registered=[ast.unparse(member) for member in next(node for node in ast.walk(extension) if isinstance(node,ast.Return)).value.elts]
 selected=[];price_expression=next(keyword.value for node in tree.body if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='PRICE_BADGE_VIDEO' for keyword in node.value.keywords if keyword.arg=='expr');price_expression=ast.literal_eval(price_expression)
 for node in tree.body:
  if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):selected.append(copy.deepcopy(node))
  elif isinstance(node,ast.AsyncFunctionDef) and node.name=='upload_image_to_pixverse':selected.append(copy.deepcopy(node))
  elif isinstance(node,ast.ClassDef) and node.name in registered:
   method=next(copy.deepcopy(member) for member in node.body if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)) and member.name=='execute');schema=next(call for call in ast.walk(node) if isinstance(call,ast.Call) and ast.unparse(call.func)=='IO.Schema');fields={keyword.arg:keyword.value for keyword in schema.keywords};defaults={argument.arg:ast.literal_eval(value) for argument,value in zip(method.args.args[-len(method.args.defaults):],method.args.defaults)} if method.args.defaults else {}
   contracts[node.name]={'arguments':[argument.arg for argument in method.args.args if argument.arg!='cls'],'defaults':defaults,'schema':ast.unparse(schema),'hidden':ast.unparse(fields['hidden']) if 'hidden' in fields else None,'outputs':ast.unparse(fields['outputs'])};selected.append(ast.ClassDef(name=node.name,bases=[],keywords=[],body=[method],decorator_list=[]))
 exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<actual-source-nodes>','exec'),namespace)
 cases=[];errors=[];response_errors=[]
 bases={'PixverseTextToVideoNode':{'prompt':'  source prompt  ','aspect_ratio':'16:9','quality':'540p','duration_seconds':5,'motion_mode':'normal','seed':0},'PixverseImageToVideoNode':{'image':TensorProbe(),'prompt':'','quality':'540p','duration_seconds':5,'motion_mode':'normal','seed':0},'PixverseTransitionVideoNode':{'first_frame':TensorProbe('first'),'last_frame':TensorProbe('last'),'prompt':'','quality':'540p','duration_seconds':5,'motion_mode':'normal','seed':0}}
 async def observe(node,arguments,failure=False,response=None):
  requests.clear();media.clear();polls.clear();downloads.clear();override.clear();override.update(response or {})
  try:
   result=namespace[node].execute(**copy.deepcopy(arguments))
   if hasattr(result,'__await__'):result=await result
  except Exception as error:
   if not failure:raise
   (response_errors if response else errors).append({'input':{'node':node,'arguments':serial(arguments)},'message':str(error),'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),'polls':copy.deepcopy(polls),'downloads':copy.deepcopy(downloads),**({'response_override':response} if response else {})})
  else:
   if failure:raise ValueError('Expected source failure '+node)
   cases.append({'input':{'node':node,'arguments':serial(arguments)},'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),'polls':copy.deepcopy(polls),'downloads':copy.deepcopy(downloads),'outputs':serial(result)})
 for template in namespace['pixverse_templates']:await observe('PixverseTemplateNode',{'template':template})
 await observe('PixverseTemplateNode',{'template':'unknown'},True)
 for node,base in bases.items():
  for quality,duration,motion in itertools.product(['360p','540p','720p','1080p'],[5,8],['normal','fast']):await observe(node,{**base,'quality':quality,'duration_seconds':duration,'motion_mode':motion,'seed':2147483647})
  for prompt in ['',' ','x'*10000,None]:await observe(node,{**base,'prompt':prompt},prompt is None or prompt=='' and node=='PixverseTextToVideoNode')
  for negative in [None,'',' ','source negative']:await observe(node,{**base,'negative_prompt':negative})
  for key,value in [('quality','unknown'),('duration_seconds',9),('motion_mode','unknown')]:await observe(node,{**base,key:value},True)
  await observe(node,base,True,{'generation':{'Resp':None,'ErrMsg':'source generation denied'}})
  await observe(node,base,True,{'generation':{'Resp':{'video_id':None}}})
  for status in [6,7,8]:await observe(node,base,True,{'poll':{'Resp':{'status':status,'url':'https://media.fixture.invalid/video.mp4'}}})
  for url in [None,'']:await observe(node,base,True,{'poll':{'Resp':{'status':1,'url':url}}})
 for ratio in ['16:9','4:3','1:1','3:4','9:16']:await observe('PixverseTextToVideoNode',{**bases['PixverseTextToVideoNode'],'aspect_ratio':ratio})
 await observe('PixverseTextToVideoNode',{**bases['PixverseTextToVideoNode'],'aspect_ratio':'unknown'},True)
 for node in ['PixverseTextToVideoNode','PixverseImageToVideoNode']:
  for template in namespace['pixverse_templates'].values():await observe(node,{**bases[node],'pixverse_template':template})
 for node,imagekey in [('PixverseImageToVideoNode','image'),('PixverseTransitionVideoNode','first_frame'),('PixverseTransitionVideoNode','last_frame')]:await observe(node,{**bases[node],imagekey:TensorProbe(imagekey,0)},True)
 for node in ['PixverseImageToVideoNode','PixverseTransitionVideoNode']:
  await observe(node,bases[node],True,{'upload':{'Resp':None,'ErrMsg':'source upload denied'}})
  await observe(node,bases[node],True,{'upload':{'Resp':{'img_id':None}}})
 rates=json.loads(re.search(r'\$prices\s*:=\s*(\{.*?\});',price_expression,re.S).group(1));quotes=[]
 for duration,quality,motion in itertools.product([5,8,9],['360p','540p','720p','1080p','unknown'],['normal','fast','unknown']):quotes.append({'duration_seconds':duration,'quality':quality,'motion_mode':motion,'maximum_microunits':int(Decimal(str(rates.get(str(duration),{}).get(quality,{}).get(motion,.9)))*1000000)})
 return {'schema_version':1,'namespace':'zed.sim.provider.pixverse','scope':'Pinned source AST and Pydantic execution; PNG preparation uses injected real encoded bytes, not native tensor encoding. Download is structural wrapper probe; native allocation and real clock continuation remain gates.','source_sha256':{path:hashlib.sha256((SOURCE_ROOT/path).read_bytes()).hexdigest() for path in SOURCES},'registered_symbols':registered,'contracts':contracts,'templates':namespace['pixverse_templates'],'cases':cases,'preflight_errors':errors,'response_errors':response_errors,'price_expression':price_expression,'price_rates':rates,'price_cases':quotes,'api_enums':{name:[member.value for member in namespace[name]] for name in ['PixverseAspectRatio','PixverseQuality','PixverseDuration','PixverseMotionMode','PixverseStatus']},'api_schemas':{name:namespace[name].model_json_schema() for name in ['PixverseTextVideoRequest','PixverseImageVideoRequest','PixverseTransitionVideoRequest','PixverseImageUploadResponse','PixverseVideoResponse','PixverseGenerationStatusResponse']}}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT);parser.add_argument('--check',action='store_true');arguments=parser.parse_args();SOURCE_ROOT=arguments.source_root;encoded=(json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n').encode();destination=FOLDER/'source-contract.json'
 if arguments.check:
  if destination.read_bytes()!=encoded:raise SystemExit('Pixverse source oracle differs')
 else:destination.write_bytes(encoded)
