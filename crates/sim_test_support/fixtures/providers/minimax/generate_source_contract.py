#!/usr/bin/env python3
import argparse,ast,asyncio,copy,hashlib,inspect,itertools,json,re
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
FOLDER=Path(__file__).resolve().parent
SOURCE_ROOT=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI')
SOURCES=['sim_api_nodes/nodes_minimax.py','sim_api_nodes/apis/minimax.py','sim_api_nodes/util/client.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/upload_helpers.py','sim_api_nodes/util/download_helpers.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/_helpers.py']
class TensorProbe:
 def __init__(self,count=1):self.shape=(count,720,720,3)
def serial(value):
 if isinstance(value,TensorProbe):return {'native_slot':'image','shape':list(value.shape)}
 if isinstance(value,dict):return {key:serial(member) for key,member in value.items()}
 if isinstance(value,(list,tuple)):return [serial(member) for member in value]
 return value
def strip(module):
 for node in ast.walk(module):
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
   node.returns=None
   for argument in node.args.args+node.args.kwonlyargs:argument.annotation=None
 return ast.fix_missing_locations(module)
async def generate():
 tree=ast.parse((SOURCE_ROOT/SOURCES[0]).read_text());requests=[];media=[];polls=[];downloads=[];override={};contracts={};prices={}
 namespace={'__builtins__':__builtins__,'IO':SimpleNamespace(NodeOutput=lambda *values:values),'torch':SimpleNamespace(Tensor=TensorProbe),'ApiEndpoint':lambda path,method='GET',query_params=None:SimpleNamespace(path=path,method=method,query_params=query_params)}
 exec(compile((SOURCE_ROOT/SOURCES[1]).read_text(),'<actual-source-api>','exec'),namespace)
 async def sync(cls,endpoint,response_model,data=None,**options):
  requests.append({'path':endpoint.path,'method':endpoint.method,'query_params':endpoint.query_params,'body':data.model_dump(mode='json',exclude_none=True) if data is not None else None,'options':options})
  base={'base_resp':{'status_code':0,'status_msg':'source'}}
  if endpoint.method=='POST':base['task_id']='source-id';base.update(override.get('initial',{}))
  else:base['file']={'download_url':'https://media.fixture.invalid/primary.mp4','backup_download_url':None};base.update(override.get('file',{}))
  return response_model.model_validate(base)
 async def poll(cls,endpoint,response_model,**options):
  response=response_model.model_validate({'base_resp':{'status_code':0,'status_msg':'source'},'task_id':'source-id','status':'Success','file_id':'00042',**override.get('poll',{})})
  polls.append({'path':endpoint.path,'method':endpoint.method,'query_params':endpoint.query_params,'estimated_duration':options.get('estimated_duration'),'status':options['status_extractor'](response)});return response
 async def upload(cls,image,**options):
  media.append({'operation':'upload_images','image':serial(image),'max_images':options.get('max_images',8),'mime_type':options.get('mime_type'),'total_pixels':4194304,'wait_label':'Uploading','show_batch_index':True});return [f'https://storage.fixture.invalid/image-{index}.png' for index in range(min(image.shape[0],options.get('max_images',8)))]
 async def download(url,**options):
  downloads.append({'url':url,'timeout':options.get('timeout'),'max_retries':options.get('max_retries',5)})
  if override.get('download_failures',{}).get(url):raise ValueError('source video construction/download failure')
  return {'canonical_video_from_file':url}
 namespace.update(sync_op=sync,poll_op=poll,upload_images_to_simapi=upload,download_url_to_video_output=download)
 validators=ast.parse((SOURCE_ROOT/SOURCES[3]).read_text());exec(compile(strip(ast.Module(body=[copy.deepcopy(node) for node in validators.body if isinstance(node,ast.FunctionDef) and node.name=='validate_string'],type_ignores=[])),'<actual-source-validation>','exec'),namespace)
 registered=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='MinimaxExtension');registered=next(node for node in ast.walk(registered) if isinstance(node,ast.Return));symbols=[ast.unparse(member) for member in registered.value.elts]
 selected=[]
 for node in tree.body:
  if isinstance(node,ast.Assign) or isinstance(node,ast.AsyncFunctionDef) and node.name=='_generate_mm_video':selected.append(copy.deepcopy(node))
  if isinstance(node,ast.ClassDef) and node.name in symbols:
   method=next(member for member in node.body if isinstance(member,ast.AsyncFunctionDef) and member.name=='execute');schema=next(call for call in ast.walk(node) if isinstance(call,ast.Call) and ast.unparse(call.func)=='IO.Schema');fields={keyword.arg:keyword.value for keyword in schema.keywords};identifier=ast.literal_eval(fields['node_id']);defaults={argument.arg:ast.literal_eval(value) for argument,value in zip(method.args.args[-len(method.args.defaults):],method.args.defaults)}
   contracts[identifier]={'symbol':node.name,'arguments':[argument.arg for argument in method.args.args if argument.arg!='cls'],'defaults':defaults,'schema':ast.unparse(schema),'hidden':ast.unparse(fields['hidden']),'outputs':ast.unparse(fields['outputs'])};prices[identifier]=ast.literal_eval(next(keyword.value for keyword in fields['price_badge'].keywords if keyword.arg=='expr'));selected.append(ast.ClassDef(name=node.name,bases=[],keywords=[],body=[copy.deepcopy(method)],decorator_list=[]))
 exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<actual-source-execute>','exec'),namespace)
 cases=[];errors=[];response_errors=[]
 bases={'MinimaxTextToVideoNode':{'prompt_text':'  source prompt  '},'MinimaxImageToVideoNode':{'prompt_text':'  source prompt  ','image':TensorProbe()},'MinimaxHailuoVideoNode':{'prompt_text':'  source prompt  '}}
 async def observe(node,arguments,failure=False,response=None):
  requests.clear();media.clear();polls.clear();downloads.clear();override.clear();override.update(response or {})
  try:result=await namespace[node].execute(**copy.deepcopy(arguments))
  except Exception as error:
   if not failure:raise
   destination=response_errors if response else errors;destination.append({'input':{'node':node,'arguments':serial(arguments)},'message':str(error),'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),'polls':copy.deepcopy(polls),'downloads':copy.deepcopy(downloads),**({'response_override':response} if response else {})})
  else:
   if failure:raise ValueError('Expected source failure '+node)
   cases.append({'input':{'node':node,'arguments':serial(arguments)},'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),'polls':copy.deepcopy(polls),'downloads':copy.deepcopy(downloads),'outputs':serial(result),**({'response_override':response} if response else {})})
 for node,base in bases.items():await observe(node,base)
 for node,models in [('MinimaxTextToVideoNode',['T2V-01','T2V-01-Director']),('MinimaxImageToVideoNode',['I2V-01-Director','I2V-01','I2V-01-live'])]:
  for model in models:
   for prompt in ['', ' ', 'x'*2000]:await observe(node,{**bases[node],'model':model,'prompt_text':prompt,'seed':2**64-1})
 for resolution,duration,optimizer,image in itertools.product(['768P','1080P','768p','1080p'],[6,10],[False,True],[None,TensorProbe()]):await observe('MinimaxHailuoVideoNode',{**bases['MinimaxHailuoVideoNode'],'resolution':resolution,'duration':duration,'prompt_optimizer':optimizer,'first_frame_image':image},resolution.upper()=='1080P' and duration!=6)
 for node,base in bases.items():
  for prompt in ['x'*2001,None]:await observe(node,{**base,'prompt_text':prompt},prompt is not None or node!='MinimaxImageToVideoNode')
  await observe(node,{**base,'model':'unknown'},True)
  response={'file':{'file':{'download_url':'https://media.fixture.invalid/primary.mp4','backup_download_url':'https://media.fixture.invalid/backup.mp4'}}}
  await observe(node,base,response=response)
  response={**response,'download_failures':{'https://media.fixture.invalid/primary.mp4':True}}
  await observe(node,base,response=response)
  for response in [{'initial':{'task_id':''}},{'poll':{'file_id':None}},{'poll':{'file_id':'not-an-integer'}},{'file':{'file':{'download_url':None}}},{'initial':{'task_id':None}},{'poll':{'status':'unknown'}}]:await observe(node,base,True,response)
 await observe('MinimaxHailuoVideoNode',{'prompt_text':None,'first_frame_image':TensorProbe()})
 await observe('MinimaxHailuoVideoNode',{'prompt_text':'','resolution':'other','model':'T2V-01','duration':None,'prompt_optimizer':None})
 await observe('MinimaxImageToVideoNode',{'prompt_text':None,'image':None},True)
 await observe('MinimaxImageToVideoNode',{'prompt_text':'source','image':TensorProbe(0)},True)
 await observe('MinimaxHailuoVideoNode',{'prompt_text':'source','first_frame_image':TensorProbe(0)},True)
 expression=prices['MinimaxHailuoVideoNode'];rates=json.loads(re.search(r'\$prices\s*:=\s*(\{.*?\});',expression,re.S).group(1));quotes=[]
 for node in contracts:
  for resolution,duration in itertools.product(['768P','1080P','768p','other'],[6,10,12]):
   rate=rates.get(resolution.lower(),{}).get(str(duration),.43) if node=='MinimaxHailuoVideoNode' else .43
   quotes.append({'node':node,'resolution':resolution,'duration':duration,'maximum_microunits':int(Decimal(str(rate))*1000000)})
 polling={};client=ast.parse((SOURCE_ROOT/SOURCES[2]).read_text())
 for node in client.body:
  if isinstance(node,ast.Assign) and ast.unparse(node.targets[0]) in {'COMPLETED_STATUSES','FAILED_STATUSES','QUEUED_STATUSES'}:polling[ast.unparse(node.targets[0])]=sorted(str(value).lower() for value in ast.literal_eval(node.value))
 return {'schema_version':1,'namespace':'zed.sim.provider.minimax','scope':'Pinned registered source AST/Pydantic request/poll/file/download branches; native uploads and VideoFromFile are injected structural probes, not native allocation. Source backup catches construction or download exceptions; real cancellation/worker-loss are terminal owner boundaries.','source_sha256':{path:hashlib.sha256((SOURCE_ROOT/path).read_bytes()).hexdigest() for path in SOURCES},'registered_symbols':symbols,'excluded_symbols':{'MinimaxSubjectToVideoNode':'explicitly commented out of MinimaxExtension.get_node_list; no executable claim'},'contracts':contracts,'cases':cases,'preflight_errors':errors,'response_errors':response_errors,'price_expressions':prices,'price_rates':rates,'price_cases':quotes,'polling':polling,'api_schemas':{name:namespace[name].model_json_schema() for name in ['MinimaxVideoGenerationRequest','MinimaxVideoGenerationResponse','MinimaxTaskResultResponse','MinimaxFileRetrieveResponse']},'api_models':[member.value for member in namespace['MiniMaxModel']]}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT);parser.add_argument('--check',action='store_true');arguments=parser.parse_args();SOURCE_ROOT=arguments.source_root
 encoded=(json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n').encode();destination=FOLDER/'source-contract.json'
 if arguments.check:
  if destination.read_bytes()!=encoded:raise SystemExit('MiniMax source oracle differs')
 else:destination.write_bytes(encoded)
