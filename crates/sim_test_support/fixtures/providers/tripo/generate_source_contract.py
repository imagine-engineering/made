"""Independent pinned Tripo nodes/API/helpers; network and app owners remain separate."""
import argparse,ast,asyncio,copy,hashlib,json,math,os,shutil,sys,tempfile,time
from enum import Enum
from io import BytesIO
from pathlib import Path
from types import ModuleType,SimpleNamespace
import numpy as np,torch
from PIL import Image
ROOT=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI');DEST=Path(__file__).with_name('source-contract.json')
FILES=['sim_api_nodes/nodes_tripo.py','sim_api_nodes/apis/tripo.py','sim_api_nodes/util/upload_helpers.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/download_helpers.py','sim_api/latest/_util/geometry_types.py','sim/utils.py']
NODES=['TripoConversionNode','TripoImageToModelNode','TripoImportModelNode','TripoMultiviewToModelNode','TripoP1ImageToModelNode','TripoP1MultiviewToModelNode','TripoP1TextToModelNode','TripoRefineNode','TripoRetargetNode','TripoRigNode','TripoTextToModelNode','TripoTextureNode'];torch.set_num_threads(1)
GLB=ROOT.parent/'ComfyUI-Frontend/browser_tests/assets/workflowInMedia/workflow.glb';OBJ=ROOT.parent/'ComfyUI-Frontend/browser_tests/assets/cube.obj'
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
 requests=[];constructors=[];polls=[];uploads=[];downloads=[];saves=[];events=[];options={};uuid_number=0
 namespace={'__builtins__':__builtins__,'torch':torch,'np':np,'Image':Image,'BytesIO':BytesIO,'Path':Path,'math':math,'os':os,'time':time,'shutil':shutil,'IO':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *values:values),'Input':SimpleNamespace(),'uuid':None,'mimetype_to_extension':lambda mime:mime.split('/')[-1].lower()}
 selected(FILES[5],{'File3D'},namespace,True);namespace['Types']=SimpleNamespace(File3D=namespace['File3D'])
 api=ModuleType('pinned_tripo_api');sys.modules[api.__name__]=api;exec(compile((ROOT/FILES[1]).read_text(),str(ROOT/FILES[1]),'exec'),api.__dict__);namespace.update(api.__dict__)
 def uuid4():
  nonlocal uuid_number
  uuid_number+=1;return f'00000000-0000-4000-8000-{uuid_number:012d}'
 namespace['uuid']=SimpleNamespace(uuid4=uuid4)
 selected(FILES[6],{'common_upscale'},namespace)
 selected(FILES[3],{'tensor_to_bytesio','tensor_to_pil','pil_to_bytesio','downscale_image_tensor','_compute_downscale_dims'},namespace)
 for assignment in ast.parse((ROOT/FILES[2]).read_text()).body:
  if isinstance(assignment,ast.Assign):
   try:value=ast.literal_eval(assignment.value)
   except (ValueError,TypeError):continue
   for target in assignment.targets:
    if isinstance(target,ast.Name):namespace[target.id]=value
 async def upload(cls,stream,filename,mime,wait_label='Uploading',progress_origin_ts=None):
  index=len(uploads);uploads.append({'file_name':filename,'mime_type':mime,'wait_label':wait_label,'bytes_hex':stream.getvalue().hex()});events.append({'upload':index});return f'https://source.fixture.invalid/input-{index+1}.bin'
 namespace['upload_file_to_simapi']=upload;selected(FILES[2],{'upload_images_to_simapi','upload_3d_model_to_simapi'},namespace)
 async def sync(cls,endpoint,response_model,data=None,**kwargs):
  payload={'code':options.get('code',0),'data':{'task_id':options.get('task_id','source-task-42')}}
  if options.get('missing_task_id'):payload['data']={}
  record={'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(mode='json',exclude_none=True),'options':kwargs,'response':copy.deepcopy(payload)};requests.append(record);events.append({'request':endpoint.path})
  response=response_model.model_validate(payload);record['normalized_response']=response.model_dump(mode='json',exclude_none=True);return response
 async def poll(cls,poll_endpoint,response_model,**kwargs):
  record={'path':poll_endpoint.path,'method':poll_endpoint.method,'options':{key:normalize(value) for key,value in kwargs.items() if not callable(value)},'responses':[],'statuses':[]};polls.append(record);events.append({'poll':poll_endpoint.path})
  for status,progress in [('queued',0),('running',50),(options.get('poll_status','success'),100)]:
   output=options.get('output',{'pbr_model':'https://source.fixture.invalid/pbr.glb','model':'https://source.fixture.invalid/model.glb','base_model':'https://source.fixture.invalid/base.glb'})
   payload={'code':0,'data':{'task_id':options.get('poll_task_id','source-task-42'),'status':status,'progress':options.get('progress',progress),'consumed_credit':options.get('credits',7),'output':output}}
   record['responses'].append({'raw':copy.deepcopy(payload)})
   response=response_model.model_validate(payload);record['responses'][-1]['normalized']=response.model_dump(mode='json',exclude_none=True)
   record['statuses'].append({'status':normalize(kwargs['status_extractor'](response)),'progress':kwargs['progress_extractor'](response),'price':kwargs['price_extractor'](response) if 'price_extractor' in kwargs else None})
  return response
 async def download(url,dest,**kwargs):downloads.append({'url':url,'options':kwargs,'bytes_hex':GLB.read_bytes().hex()});events.append({'download':url});dest.write(GLB.read_bytes());dest.seek(0)
 namespace.update(sync_op=sync,poll_op=poll,download_url_to_bytesio=download,ApiEndpoint=lambda path,method='GET':SimpleNamespace(path=path,method=method))
 selected(FILES[4],{'download_url_to_file_3d'},namespace)
 selected(FILES[0],{'get_model_url_from_response','poll_until_finished','_resolve_p1_texture_fields','_build_p1_request_kwargs'},namespace)
 selected(FILES[0],set(NODES),namespace,True)
 def normalize(value):
  if isinstance(value,Enum):return value.value
  if hasattr(value,'model_dump'):return value.model_dump(mode='json',exclude_none=False)
  if isinstance(value,namespace['File3D']):return {'format':value.format,'bytes_hex':value.get_bytes().hex(),'is_disk_backed':value.is_disk_backed}
  if isinstance(value,(tuple,list)):return [normalize(item) for item in value]
  if isinstance(value,dict):return {key:normalize(item) for key,item in value.items()}
  return value
 for name in [name for name in namespace if name.startswith('Tripo') and name.endswith('Request')]:
  original=namespace[name]
  def constructor(*args,_name=name,_original=original,**kwargs):
   record={'model':_name,'raw':normalize(kwargs)};constructors.append(record)
   result=_original(*args,**kwargs);record['normalized']=result.model_dump(mode='json',exclude_none=True);return result
  namespace[name]=constructor
 def base(node):
  settings={}
  if 'TextToModel' in node:settings['prompt']=' source scene '
  if 'P1' in node:settings['output_mode']={'output_mode':'Geometry only'}
  if node in ['TripoTextureNode','TripoRefineNode']:settings['model_task_id']='original-task-17'
  if node in ['TripoConversionNode','TripoRigNode','TripoRetargetNode']:settings['original_model_task_id']='original-task-17'
  if node=='TripoRetargetNode':settings['animation']='preset:walk'
  if node=='TripoConversionNode':settings.update(format='FBX',quad=False,force_symmetry=False,face_limit=-1,flatten_bottom=False,flatten_bottom_threshold=0.,texture_size=4096,texture_format='JPEG',pivot_to_center_bottom=False,scale_factor=1.,with_animation=False,pack_uv=False,bake=False,part_names='',fbx_preset='blender',export_vertex_colors=False,export_orientation='default',animate_in_place=False)
  images={}
  if 'ImageToModel' in node or 'Multiview' in node:images['image']={'batch':1,'width':2,'height':2,'channels':3}
  if 'Multiview' in node:images['image_left']={'batch':1,'width':2,'height':2,'channels':4}
  return {'node':node,'settings':settings,'images':images,'file':{'format':'glb','stream_kind':'memory'} if node=='TripoImportModelNode' else None}
 def arguments(value):
  result=copy.deepcopy(value['settings'])
  for ordinal,(name,descriptor) in enumerate(value['images'].items()):
   if descriptor is None:result[name]=None;continue
   image=torch.empty((descriptor['batch'],descriptor['height'],descriptor['width'],descriptor['channels']),dtype=torch.float32)
   for batch in range(descriptor['batch']):
    for channel in range(descriptor['channels']):image[batch,...,channel]=[.5,.25,.125,1.][channel]+((ordinal+batch)*.01 if channel<3 else 0.)
   result[name]=image
  if value['node']=='TripoImportModelNode':
   file=value['file'];asset=OBJ if file.get('format','').lstrip('.').lower()=='obj' else GLB
   if file.get('oversized'):source=BytesIO(asset.read_bytes()+bytes(150*1024*1024+1-asset.stat().st_size))
   elif file['stream_kind']=='disk':source=str(asset)
   else:source=BytesIO(asset.read_bytes());source.seek(3)
   result['model_3d']=namespace['File3D'](source=source,file_format=file.get('format',''))
  return result
 cases=[];errors=[]
 with tempfile.TemporaryDirectory(prefix='sim-tripo-source-save-') as output_directory:
  namespace['get_output_directory']=lambda:output_directory
  async def observe(value):
   nonlocal uuid_number
   uuid_number=0
   for collection in [requests,constructors,polls,uploads,downloads,saves,events]:collection.clear()
   for path in Path(output_directory).iterdir():path.unlink()
   options.clear();options.update(value.get('oracle_options',{}))
   try:result=await namespace[value['node']].execute(**arguments(value))
   except (NameError,ImportError,TypeError) as error:raise RuntimeError(f'harness failure: {value["node"]}: {error}') from error
   except AttributeError as error:
    if "TripoTaskResponse' object has no attribute 'error'" not in str(error):raise
    errors.append({'input':copy.deepcopy(value),'error_type':type(error).__name__,'error_message':str(error),'requests':copy.deepcopy(requests),'constructors':copy.deepcopy(constructors),'polls':copy.deepcopy(polls),'uploads':copy.deepcopy(uploads),'downloads':copy.deepcopy(downloads),'saves':[],'events':copy.deepcopy(events)});return
   except Exception as error:errors.append({'input':copy.deepcopy(value),'error_type':type(error).__name__,'error_message':str(error),'requests':copy.deepcopy(requests),'constructors':copy.deepcopy(constructors),'polls':copy.deepcopy(polls),'uploads':copy.deepcopy(uploads),'downloads':copy.deepcopy(downloads),'saves':[],'events':copy.deepcopy(events)});return
   for path in Path(output_directory).iterdir():saves.append({'file_name':path.name,'bytes_hex':path.read_bytes().hex()});events.append({'save':path.name})
   cases.append({'input':copy.deepcopy(value),'requests':copy.deepcopy(requests),'constructors':copy.deepcopy(constructors),'polls':copy.deepcopy(polls),'uploads':copy.deepcopy(uploads),'downloads':copy.deepcopy(downloads),'saves':copy.deepcopy(saves),'events':copy.deepcopy(events),'output':normalize(result)})
  for node in NODES:await observe(base(node))
  for node in ['TripoTextToModelNode','TripoP1TextToModelNode']:
   for prompt in ['', ' ', 'a'*1024,'a'*1025]:value=base(node);value['settings']['prompt']=prompt;await observe(value)
   for negative in [None,'','x'*255,'x'*256,'x'*1024,'x'*1025]:value=base(node);value['settings']['negative_prompt']=negative;await observe(value)
  for node in ['TripoTextToModelNode','TripoImageToModelNode']:
   for style in ['None','gold','person:person2cartoon','invalid']:value=base(node);value['settings']['style']=style;await observe(value)
   for model in [None,'v3.1-20260211','v1.4-20240625','invalid']:value=base(node);value['settings']['model_version']=model;await observe(value)
   for settings in [{'texture':False,'pbr':True},{'texture':True,'pbr':False},{'face_limit':-1},{'face_limit':0},{'face_limit':42,'quad':True},{'model_seed':-1,'texture_seed':4294967296},{'texture_quality':'detailed','geometry_quality':'detailed'}]:value=base(node);value['settings'].update(settings);await observe(value)
  for node in ['TripoImageToModelNode','TripoP1ImageToModelNode']:
   for image in [None,{'batch':0,'width':2,'height':2,'channels':3},{'batch':2,'width':2,'height':2,'channels':4}]:value=base(node);value['images']['image']=image;await observe(value)
  for node in ['TripoMultiviewToModelNode','TripoP1MultiviewToModelNode']:
   for present in [('image',),('image_left','image_back'),('image','image_back'),('image','image_right'),('image','image_left','image_back','image_right')]:value=base(node);value['images']={name:({'batch':1,'width':2,'height':2,'channels':3} if name in present else None) for name in ['image','image_left','image_back','image_right']};value['settings']['quad']=True if node=='TripoMultiviewToModelNode' else None;value['settings'].pop('quad',None) if node=='TripoP1MultiviewToModelNode' else None;await observe(value)
  for node in [item for item in NODES if 'P1' in item]:
   for output in [{'output_mode':'Textured'},{'output_mode':'Textured','pbr':False,'texture_quality':'detailed','texture_seed':42},{'output_mode':'Unknown','pbr':None},{'output_mode':'Geometry only','pbr':True}]:value=base(node);value['settings']['output_mode']=output;value['settings'].update(auto_size=True,export_uv=False,compress_geometry=True);await observe(value)
   for face in [-1,47,48,20000,20001]:value=base(node);value['settings']['face_limit']=face;await observe(value)
  for prompt in ['', ' \x1c paint \x1f ', '   ']:value=base('TripoTextureNode');value['settings']['texture_prompt']=prompt;await observe(value)
  for animation in ['preset:idle','preset:quadruped:walk','invalid']:value=base('TripoRetargetNode');value['settings']['animation']=animation;await observe(value)
  for format in ['GLTF','USDZ','FBX','OBJ','STL','3MF','GLB']:value=base('TripoConversionNode');value['settings']['format']=format;await observe(value)
  for names in ['', ' , , ',' head, \x1c arm ,tail ']:value=base('TripoConversionNode');value['settings']['part_names']=names;await observe(value)
  value=base('TripoConversionNode');value['settings'].update(quad=True,force_symmetry=True,face_limit=123,flatten_bottom=True,flatten_bottom_threshold=.25,texture_size=2048,texture_format='PNG',pivot_to_center_bottom=True,scale_factor=.5,with_animation=True,pack_uv=True,bake=True,fbx_preset='mixamo',export_vertex_colors=True,export_orientation='align_image',animate_in_place=True);await observe(value)
  for format in ['.GLB','...GlB','obj','gltf','unknown',' GLB','']:
   for stream in ['memory','disk']:value=base('TripoImportModelNode');value['file'].update(format=format,stream_kind=stream);await observe(value)
  value=base('TripoImportModelNode');value['file']['oversized']=True;await observe(value)
  for node in NODES:
   for option in [{'code':3},{'missing_task_id':True},{'progress':101},{'poll_status':'failed'},{'credits':0},{'credits':-1},{'output':{'model':'https://source.fixture.invalid/model-only.glb'}},{'output':{'base_model':'https://source.fixture.invalid/base-only.glb'}},{'output':{}},{'poll_task_id':'different-final-id'}]:value=base(node);value['oracle_options']=option;await observe(value)
 schema={}
 for original in ast.parse((ROOT/FILES[0]).read_text()).body:
  if isinstance(original,ast.ClassDef) and original.name in NODES:
   schema[original.name]=[ast.unparse(member) for member in original.body if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)) and member.name=='define_schema']
 return {'sources':{file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in FILES},'assets':{'glb_sha256':hashlib.sha256(GLB.read_bytes()).hexdigest(),'obj_sha256':hashlib.sha256(OBJ.read_bytes()).hexdigest()},'versions':{'torch':torch.__version__,'pillow':Image.__version__},'cases':cases,'errors':errors,'schema_facts':schema,'scope':'actual node/API/image upload/3D File3D/upload/download compatibility save helpers; fixed hermetic transport/poll owner, no generic poll timing, native ABI or app filesystem permission proof'}
parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();result=asyncio.run(generate());encoded=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
if args.check:
 if DEST.read_text()!=encoded:raise SystemExit('source oracle drift')
else:DEST.write_text(encoded)
print(f"Tripo actual source boundary: {len(result['cases'])} successes,{len(result['errors'])} ordered failures")
