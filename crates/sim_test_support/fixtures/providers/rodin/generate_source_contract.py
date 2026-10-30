import argparse,ast,asyncio,contextlib,copy,hashlib,io,json,logging,math,os,re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import aiohttp
import numpy as np
from PIL import Image
FOLDER=Path(__file__).resolve().parent
SOURCE_ROOT=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI')
FRONTEND_ROOT=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI-Frontend')
SOURCES=['sim_api_nodes/nodes_rodin.py','sim_api_nodes/apis/rodin.py','sim_api_nodes/util/client.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/download_helpers.py','sim_api/latest/_util/geometry_types.py']
class TensorProbe:
 def __init__(self,slot='image',count=1,batched=True,channels=3):self.slot=slot;self.shape=(count,1,1,channels) if batched else (1,1,channels);self.channels=channels
 def __getitem__(self,index):
  if index>=self.shape[0]:raise IndexError('source image index')
  return TensorProbe(f'{self.slot}:{index}',batched=False,channels=self.channels)
 def cpu(self):return self
 def numpy(self):return np.array([[[.5,.25,.125]+([.375] if self.channels==4 else [])]],dtype=np.float32)
def serial(value):
 if isinstance(value,TensorProbe):return {'native_slot':value.slot,'shape':list(value.shape),'pixels':[.5,.25,.125]+([.375] if value.channels==4 else [])}
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
 logging.disable(logging.CRITICAL);requests=[];media=[];downloads=[];writes=[];delays=[];progress=[];overrides={};contracts={};prices={}
 assets={'glb':(FRONTEND_ROOT/'browser_tests/assets/workflowInMedia/workflow.glb').read_bytes(),'obj':(FRONTEND_ROOT/'browser_tests/assets/cube.obj').read_bytes()}
 namespace={'__builtins__':__builtins__,'__name__':'rodin_source_probe','IO':SimpleNamespace(NodeOutput=lambda *values:values),'BytesIO':io.BytesIO,'Image':Image,'np':np,'math':math,'logging':logging,'aiohttp':aiohttp,'json':json,'contextlib':contextlib,'dataclass':dataclass,'asyncio':asyncio,'time':SimpleNamespace(monotonic=lambda:0.0),'is_processing_interrupted':lambda:False}
 exec(compile((SOURCE_ROOT/SOURCES[1]).read_text(),'<actual-source-api>','exec'),namespace)
 geometry=ast.parse((SOURCE_ROOT/SOURCES[5]).read_text());exec(compile(strip(ast.Module(body=[copy.deepcopy(node) for node in geometry.body if isinstance(node,ast.ClassDef) and node.name=='File3D'],type_ignores=[])),'<actual-source-File3D-wrapper>','exec'),namespace)
 class StorageProbe(io.BytesIO):
  def __init__(self,path,mode):super().__init__();self.path=path;self.mode=mode
  def close(self):
   if not self.closed:writes.append({'path':self.path,'bytes':len(self.getvalue()),'sha256':hashlib.sha256(self.getvalue()).hexdigest()})
   super().close()
 namespace.update(os=SimpleNamespace(path=os.path,makedirs=lambda *args,**options:None),sim_paths=SimpleNamespace(get_output_directory=lambda:'/source-output'),open=lambda path,mode:StorageProbe(path,mode),Types=SimpleNamespace(File3D=namespace['File3D']))
 async def download_bytes(url,dest,**options):
  data=assets['glb'] if url.endswith('.glb') else assets['obj'];downloads.append({'kind':'bytes','url':url,'destination':dest if isinstance(dest,str) else 'BytesIO','options':options})
  if isinstance(dest,str):writes.append({'path':dest,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
  else:dest.write(data);dest.seek(0)
 download_source=ast.parse((SOURCE_ROOT/SOURCES[4]).read_text());selected=[copy.deepcopy(node) for node in download_source.body if isinstance(node,ast.AsyncFunctionDef) and node.name=='download_url_to_file_3d'];exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<actual-source-file-download-wrapper>','exec'),namespace);actual_download=namespace['download_url_to_file_3d']
 async def download_file(url,file_format,**options):downloads.append({'kind':'file3d','url':url,'format':file_format,'options':options});return await actual_download(url,file_format,**options)
 namespace.update(download_url_to_bytesio=download_bytes,download_url_to_file_3d=download_file,ApiEndpoint=lambda path,method='GET',**options:SimpleNamespace(path=path,method=method,**options))
 client=ast.parse((SOURCE_ROOT/SOURCES[2]).read_text());client_nodes=[]
 for node in client.body:
  if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in ['COMPLETED_STATUSES','FAILED_STATUSES','QUEUED_STATUSES'] for target in node.targets):client_nodes.append(copy.deepcopy(node))
  elif isinstance(node,ast.ClassDef) and node.name=='_PollUIState':client_nodes.append(copy.deepcopy(node))
  elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ['poll_op','poll_op_raw','_normalize_statuses','_normalize_status_value','_validate_or_raise','_wrap_model_extractor']:client_nodes.append(copy.deepcopy(node))
 exec(compile(strip(ast.Module(body=client_nodes,type_ignores=[])),'<actual-source-poll-client>','exec'),namespace)
 class ProgressProbe:
  def __init__(self,total):self.total=total
  def update_absolute(self,value,total):progress.append({'completed':value,'total':total})
 class Interrupted(Exception):pass
 class NetworkError(Exception):pass
 class ApiError(Exception):pass
 async def delay(seconds,*arguments):delays.append(seconds)
 namespace.update(utils=SimpleNamespace(ProgressBar=ProgressProbe),sleep_with_interrupt=delay,_display_time_progress=lambda *args,**kwargs:None,ProcessingInterrupted=Interrupted,LocalNetworkError=NetworkError,ApiServerError=ApiError)
 validation=ast.parse((SOURCE_ROOT/SOURCES[3]).read_text());exec(compile(strip(ast.Module(body=[copy.deepcopy(node) for node in validation.body if isinstance(node,ast.FunctionDef) and node.name=='validate_string'],type_ignores=[])),'<source-validation>','exec'),namespace)
 tree=ast.parse((SOURCE_ROOT/SOURCES[0]).read_text());registered=['Rodin3D_Regular','Rodin3D_Detail','Rodin3D_Smooth','Rodin3D_Sketch','Rodin3D_Gen2','Rodin3D_Gen25_Image','Rodin3D_Gen25_Text'];selected=[]
 for node in tree.body:
  if isinstance(node,(ast.Assign,ast.AnnAssign)) and '_PRICE_EXPR' in ast.unparse(node).split('=')[0] or isinstance(node,(ast.Assign,ast.AnnAssign)) and any(name in ast.unparse(node).split('=')[0] for name in ['_QUALITY_MESH_OPTIONS','_PREVIEWABLE_3D_EXTS','_MODE_REGULAR','_MODE_FAST','_MODE_EXTREME_HIGH']):selected.append(copy.deepcopy(node))
  elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ['get_quality_mode','tensor_to_filelike','create_generate_task','check_rodin_status','extract_progress','poll_for_task_status','get_rodin_download_list','download_files','_rodin_multipart_parser','_create_gen25_task','_download_gen25_files','_resolve_mode_params','_build_request']:selected.append(copy.deepcopy(node))
  elif isinstance(node,ast.ClassDef) and node.name in registered:
   method=next(copy.deepcopy(member) for member in node.body if isinstance(member,ast.AsyncFunctionDef) and member.name=='execute');schema=next(call for call in ast.walk(node) if isinstance(call,ast.Call) and ast.unparse(call.func)=='IO.Schema');fields={keyword.arg:keyword.value for keyword in schema.keywords};contracts[node.name]={'arguments':[argument.arg for argument in method.args.args if argument.arg!='cls'],'defaults':{},'schema':ast.unparse(schema),'hidden':ast.unparse(fields['hidden']),'outputs':ast.unparse(fields['outputs'])};expr=next(keyword.value for keyword in fields['price_badge'].keywords if keyword.arg=='expr');prices[node.name]=ast.literal_eval(expr) if isinstance(expr,ast.Constant) else ast.unparse(expr);selected.append(ast.ClassDef(name=node.name,bases=[],keywords=[],body=[method],decorator_list=[]))
 exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<actual-source-node-and-helper>','exec'),namespace);actual_tensor=namespace['tensor_to_filelike']
 def tensor_file(tensor):
  encoded=actual_tensor(tensor);data=encoded.getvalue();media.append({'operation':'tensor_to_filelike','image':serial(tensor),'mime_type':'image/png','total_pixels':4194304,'resampling':'PIL.LANCZOS','bytes_hex':data.hex()});return encoded
 namespace['tensor_to_filelike']=tensor_file
 client_request=next(node for node in client.body if isinstance(node,ast.AsyncFunctionDef) and node.name=='_request_base');multipart=next(node for node in ast.walk(client_request) if isinstance(node,ast.If) and ast.unparse(node.test)=="cfg.content_type == 'multipart/form-data' and method != 'GET'");multipart_code=compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(multipart)],type_ignores=[])),'<actual-source-client-multipart>','exec')
 async def form_wire(data,files,parser):
  local={'cfg':SimpleNamespace(content_type='multipart/form-data',data=data,files=files,multipart_parser=parser),'method':'POST','payload_headers':{},'payload_kw':{},'aiohttp':aiohttp,'BytesIO':io.BytesIO,'contextlib':contextlib};exec(multipart_code,local);form=local['payload_kw']['data'];payload=form();payload._boundary=b'rodin-source-boundary'
  class Writer:
   def __init__(self):self.data=bytearray()
   async def write(self,value):self.data.extend(value)
  writer=Writer();await payload.write(writer);return writer.data.hex()
 generate_response={'uuid':'task-1','jobs':{'uuids':['job-1'],'subscription_key':'sub-1'}};status_response={'jobs':[{'uuid':'job-1','status':'Done'}]};download_response={'list':[{'url':'https://fixture.invalid/model.glb','name':'model.glb'},{'url':'https://fixture.invalid/cube.obj','name':'cube.obj'}]}
 async def sync(cls,endpoint,response_model,data=None,files=None,**options):
  body=data.model_dump(mode='json',exclude_none=True);wire=await form_wire(body,files,options.get('multipart_parser')) if options.get('content_type')=='multipart/form-data' else None;requests.append({'path':endpoint.path,'method':endpoint.method,'body':body,'options':{key:(value.__name__ if callable(value) else value) for key,value in options.items()},**({'multipart_body_hex':wire} if wire is not None else {})});response=overrides.get('generate',generate_response) if endpoint.path.endswith('/rodin') else overrides.get('download',download_response);return response_model.model_validate(response)
 async def sync_raw(cls,endpoint,data=None,**options):
  requests.append({'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(mode='json',exclude_none=True),'options':options});return copy.deepcopy(overrides.get('status',status_response))
 namespace.update(sync_op=sync,sync_op_raw=sync_raw)
 bases={}
 for node in registered[:5]:bases[node]={'Images':TensorProbe('Images',2),'Seed':0,**({'Material_Type':'PBR','Polygon_count':'18K-Quad'} if node!='Rodin3D_Sketch' else {}),**({'TAPose':False} if node=='Rodin3D_Gen2' else {})}
 common={'mode':{'mode':'Regular','tier':'Gen-2.5-High','polygon_count':'Default','creative':False},'material':'Shaded','geometry_file_format':'glb','texture_mode':'Default','seed':0,'TAPose':False,'hd_texture':False,'texture_delight':False,'addon_highpack':False,'bbox_width':0,'bbox_height':0,'bbox_length':0,'height_cm':0}
 bases['Rodin3D_Gen25_Image']={**common,'images':{'image9':TensorProbe('first',2),'image2':TensorProbe('second',1)},'use_original_alpha':False};bases['Rodin3D_Gen25_Text']={**common,'prompt':' source model '}
 cases=[];errors=[];response_errors=[]
 def input_value(node,arguments):
  value=serial(arguments)
  if 'images' in arguments:value['images']={'ordered_entries':[{'key':key,'image':serial(image)} for key,image in arguments['images'].items()]}
  return {'node':node,'arguments':value}
 async def observe(node,args,failure=False,override=None):
  requests.clear();media.clear();downloads.clear();writes.clear();delays.clear();progress.clear();overrides.clear();overrides.update(override or {})
  try:result=await namespace[node].execute(**copy.deepcopy(args))
  except Exception as error:
   if not failure:raise
   (response_errors if override else errors).append({'input':input_value(node,args),'message':re.sub(r'0x[0-9a-fA-F]+','<address>',str(error)),'error_type':type(error).__name__,'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),'downloads':copy.deepcopy(downloads),**({'override':override} if override else {})})
  else:
   if failure:raise ValueError('Expected source failure '+node)
   outputs=[{'format':value.format,'bytes':len(value.get_bytes()),'sha256':hashlib.sha256(value.get_bytes()).hexdigest(),'source_wrapper':'actual File3D byte wrapper; not native ABI'} if isinstance(value,namespace['File3D']) else value for value in result];cases.append({'input':input_value(node,args),'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),'downloads':copy.deepcopy(downloads),'writes':copy.deepcopy(writes),'delays':copy.deepcopy(delays),'progress':copy.deepcopy(progress),'outputs':outputs,**({'override':override} if override else {})})
 for node,base in bases.items():
  await observe(node,base)
  if node in registered[:5]:
   for count in [0,1,5,6]:await observe(node,{**base,'Images':TensorProbe('Images',count)},count>5)
   if 'Polygon_count' in base:
    for label in namespace['_QUALITY_MESH_OPTIONS']|{'unknown':None}:await observe(node,{**base,'Polygon_count':label,'Seed':-2,'Material_Type':'custom',**({'TAPose':True} if node=='Rodin3D_Gen2' else {})})
  else:
   for mode in [{'mode':'Regular','tier':'custom','polygon_count':'unknown','creative':True},{'mode':'Fast','tier':'Gen-2.5-Low','mesh_faces':1234.9},{'mode':'Extreme-High','mesh_mode':'Raw','mesh_faces':2000000,'is_micro':True,'creative':True}]:await observe(node,{**base,'mode':mode,'texture_mode':'high','seed':-2,'TAPose':True,'hd_texture':True,'texture_delight':True,'addon_highpack':True,'bbox_width':1,'bbox_height':2,'bbox_length':3,'height_cm':-5})
   for mode in [{},{'mode':'unknown'},{'mode':'Fast','tier':'Gen-2.5-Low'},{'mode':'Regular'}]:await observe(node,{**base,'mode':mode},True)
   for geometry in ['obj','.OBJ','fbx','usdz','custom']:await observe(node,{**base,'geometry_file_format':geometry})
   for bbox in [(1,0,3),(0,0,0),(-1,2,3)]:await observe(node,{**base,'bbox_width':bbox[0],'bbox_height':bbox[1],'bbox_length':bbox[2]})
  for override in [{'generate':{}},{'generate':{'uuid':'','jobs':{'uuids':[],'subscription_key':''}}},{'download':{'list':[]}},{'download':{'list':[{'url':'https://fixture.invalid/cube.obj','name':'cube.obj'}]}}]:await observe(node,base,override.get('generate') is not None and (not override['generate'] or node in registered[5:]),override)
 for count in [0,1,5,6]:await observe('Rodin3D_Gen25_Image',{**bases['Rodin3D_Gen25_Image'],'images':{'image9':TensorProbe('batch',count)}},count>5)
 for refs in [{},{'image1':None}]:await observe('Rodin3D_Gen25_Image',{**bases['Rodin3D_Gen25_Image'],'images':refs},True)
 for prompt in ['',None,' ','\u001c source \u001f','α'*2500,'😀'*2501]:await observe('Rodin3D_Gen25_Text',{**bases['Rodin3D_Gen25_Text'],'prompt':prompt},prompt is None or not prompt.strip() or len(prompt.strip())>2500)
 poll_cases=[]
 for jobs in [[],[{'uuid':'one','status':'Waiting'}],[{'uuid':'one','status':'Generating'}],[{'uuid':'one','status':'Failed'}],[{'uuid':'one','status':'Done'},{'uuid':'two','status':'Generating'}],[{'uuid':'one','status':'unknown'}]]:
  requests.clear();delays.clear();progress.clear();overrides.clear();overrides['status']={'jobs':jobs}
  try:response=await namespace['poll_op'](None,namespace['ApiEndpoint']('/proxy/rodin/api/v2/status','POST'),response_model=namespace['Rodin3DCheckStatusResponse'],data=namespace['Rodin3DCheckStatusRequest'](subscription_key='sub-1'),status_extractor=namespace['check_rodin_status'],progress_extractor=namespace['extract_progress'])
  except Exception as error:poll_cases.append({'response':{'jobs':jobs},'message':str(error),'requests':len(requests),'delays':delays.copy(),'progress':progress.copy()})
  else:poll_cases.append({'response':{'jobs':jobs},'requests':len(requests),'delays':delays.copy(),'progress':progress.copy(),'result':response.model_dump(mode='json')})
 return {'schema_version':1,'namespace':'zed.sim.provider.rodin','scope':'Actual node/helper/client polling AST/Pydantic/PIL/multipart and File3D byte wrapper. NumPy input and virtual filesystem stand in for canonical native preparation/storage; no native allocation/codec/publication/live calls.','source_sha256':{path:hashlib.sha256((SOURCE_ROOT/path).read_bytes()).hexdigest() for path in SOURCES},'contracts':contracts,'cases':cases,'preflight_errors':errors,'response_errors':response_errors,'poll_cases':poll_cases,'quality_options':namespace['_QUALITY_MESH_OPTIONS'],'price_expressions':prices,'gen25_price_expression':namespace['_PRICE_EXPR'],'api_schemas':{name:namespace[name].model_json_schema() for name in ['Rodin3DGenerateRequest','Rodin3DGen25Request','Rodin3DGenerateResponse','Rodin3DCheckStatusResponse','Rodin3DDownloadResponse']},'multipart_boundary':'rodin-source-boundary','aiohttp_version':aiohttp.__version__,'aiohttp_formdata_sha256':hashlib.sha256(Path(aiohttp.formdata.__file__).read_bytes()).hexdigest(),'encoded_models':{format:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'producer':'Pinned real ComfyUI-Frontend asset, actual File3D byte wrapper; no native decode'} for format,data in assets.items()}}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT);parser.add_argument('--check',action='store_true');arguments=parser.parse_args();SOURCE_ROOT=arguments.source_root;encoded=(json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n').encode();destination=FOLDER/'source-contract.json'
 if arguments.check:
  if destination.read_bytes()!=encoded:raise SystemExit('Rodin source oracle differs')
 else:destination.write_bytes(encoded)
