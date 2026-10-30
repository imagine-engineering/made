"""Actual pinned Meshy execute/Pydantic/upload-plan oracle; no native pixel golds."""
import argparse,ast,asyncio,copy,hashlib,json,math,time,tempfile,struct,shutil
from io import BytesIO
from typing import IO as BinaryIO
import torch
import numpy as np
from PIL import Image
torch.set_num_threads(1)
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI')
DEST=Path(__file__).with_name('source-contract.json')
FILES=['sim_api/latest/_util/geometry_types.py','sim_api_nodes/nodes_meshy.py','sim_api_nodes/apis/meshy.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/upload_helpers.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/client.py','sim_api_nodes/util/download_helpers.py']
NODES=['MeshyTextToModelNode','MeshyRefineNode','MeshyImageToModelNode','MeshyMultiImageToModelNode','MeshyRigModelNode','MeshyAnimateModelNode','MeshyTextureNode']
def selected(path,names,namespace,classes=False):
    nodes=[]
    for original in ast.parse((ROOT/path).read_text()).body:
        if getattr(original,'name',None) not in names:continue
        node=copy.deepcopy(original)
        if classes:node.body=[member for member in node.body if isinstance(member,ast.AsyncFunctionDef) and member.name=='execute']
        for member in ast.walk(node):
            if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
                member.returns=None
                for argument in [*member.args.args,*member.args.kwonlyargs]:argument.annotation=None
        nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(ROOT/path),'exec'),namespace)
class ImageProbe:
    def __init__(self,role,width=160,height=160,batch=1):self.role,self.shape=role,(batch,height,width,3)
    def __getitem__(self,index):
        if not 0<=index<self.shape[0]:raise IndexError(index)
        value=ImageProbe(f'{self.role}-{index}',self.shape[2],self.shape[1]);value.shape=value.shape[1:];return value
async def generate():
    uploads=[];requests=[];polls=[];downloads=[]
    namespace={'__builtins__':__builtins__,'IO':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *values:values),'Input':SimpleNamespace(Image=object),'time':SimpleNamespace(monotonic=lambda:0.)}
    api=ast.parse((ROOT/'sim_api_nodes/apis/meshy.py').read_text());api.body=[node for node in api.body if not isinstance(node,ast.ImportFrom) or node.module!='sim_api.latest'];exec(compile(api,str(ROOT/'sim_api_nodes/apis/meshy.py'),'exec'),namespace)
    selected('sim_api_nodes/util/validation_utils.py',{'validate_string'},namespace)
    def encode(image,**options):
        uploads.append({'role':image.role,'width':image.shape[1],'height':image.shape[0],'total_pixels':options.get('total_pixels'),'declared_content_type':options.get('mime_type'),'encoded_mime_type':'image/png'})
        return SimpleNamespace(name=image.role+'.png',role=image.role)
    async def upload(cls,stream,filename,mime,label,started):
        uploads[-1].update(file_name=filename,wait_label=label)
        return 'https://storage.fixture.invalid/'+stream.role+'.png'
    namespace.update(tensor_to_bytesio=encode,upload_file_to_simapi=upload)
    selected('sim_api_nodes/util/upload_helpers.py',{'upload_images_to_simapi'},namespace)
    async def sync(cls,endpoint,response_model,data,**options):
        requests.append({'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(exclude_none=True)});return response_model.model_validate({'result':'task-42'})
    async def poll(cls,endpoint,response_model,**options):
        body={'id':'task-42','type':'model','thumbnail_url':'https://output.fixture.invalid/unused.png','status':'SUCCEEDED','progress':100,'model_urls':{'glb':'https://output.fixture.invalid/model.glb','fbx':'https://output.fixture.invalid/model.fbx'},'result':{'rigged_character_glb_url':'https://output.fixture.invalid/model.glb','rigged_character_fbx_url':'https://output.fixture.invalid/model.fbx','animation_glb_url':'https://output.fixture.invalid/model.glb','animation_fbx_url':'https://output.fixture.invalid/model.fbx'}}
        result=response_model.model_validate(body)
        polls.append({'path':endpoint.path,'method':endpoint.method,'poll_interval':options.get('poll_interval',5.),'status':options['status_extractor'](result),'progress':options['progress_extractor'](result),'body':result.model_dump(exclude_none=True)})
        return result
    async def download(url,format,**options):
        value={'url':url,'format':format,'task_id':options.get('task_id'),'save_file_name':f"{options['task_id']}.{format}"};downloads.append(value);return value
    namespace.update(sync_op=sync,poll_op=poll,download_url_to_file_3d=download,ApiEndpoint=lambda path,method='GET':SimpleNamespace(path=path,method=method))
    selected('sim_api_nodes/nodes_meshy.py',set(NODES),namespace,True)
    def image(role,batch=1):return {'role':role,'width':160,'height':160,'batch':batch}
    def base(node):
        common={'model':'latest','should_remesh':{'should_remesh':'true','topology':'triangle','target_polycount':300000},'symmetry_mode':'auto','pose_mode':'A-pose','seed':42}
        settings={NODES[0]:{**common,'prompt':' ☁ ','style':'realistic'},NODES[1]:{'model':'latest','meshy_task_id':'previous-task','enable_pbr':False,'texture_prompt':''},NODES[2]:{**common,'should_texture':{'should_texture':'false'}},NODES[3]:{**common,'should_texture':{'should_texture':'false'}},NODES[4]:{'meshy_task_id':'previous-task','height_meters':1.7},NODES[5]:{'rig_task_id':'previous-rig','action_id':0},NODES[6]:{'model':'latest','meshy_task_id':'previous-task','enable_original_uv':True,'pbr':False,'text_style_prompt':' style '}}[node]
        if node==NODES[0]:settings.pop('model',None);settings['model']='latest'
        values=[]
        if node==NODES[2]:values=[image('image',2)]
        if node==NODES[3]:values=[image('imageZ',2),image('imageA',3)]
        return {'node':node,'settings':settings,'images':values}
    def kwargs(value):
        output=copy.deepcopy(value['settings']);by_role={entry['role']:ImageProbe(**entry) for entry in value['images']}
        if value['node']==NODES[2]:output['image']=by_role['image']
        if value['node']==NODES[3]:output['images']={entry['role']:by_role[entry['role']] for entry in value['images'] if entry['role'].startswith('image')}
        for role in ['texture_image','image_style']:
            if role in by_role:
                if value['node'] in [NODES[2],NODES[3]]:output['should_texture'][role]=by_role[role]
                else:output[role]=by_role[role]
        if value['node'] in [NODES[2],NODES[3]] and output['should_texture']['should_texture']=='true':output['should_texture'].setdefault('texture_image',None)
        return output
    def clear():
        for values in [uploads,requests,polls,downloads]:values.clear()
    cases=[];errors=[]
    async def observe(value):
        clear()
        try:output=await namespace[value['node']].execute(**kwargs(value))
        except Exception as error:errors.append({'input':copy.deepcopy(value),'error_type':type(error).__name__,'uploads':copy.deepcopy(uploads),'requests':len(requests)});return
        cases.append({'input':copy.deepcopy(value),'uploads':copy.deepcopy(uploads),'request':copy.deepcopy(requests[0]),'poll':copy.deepcopy(polls[0]),'downloads':copy.deepcopy(downloads),'output':list(output)})
    for node in NODES:await observe(base(node))
    facts={}
    for node in ast.parse((ROOT/'sim_api_nodes/nodes_meshy.py').read_text()).body:
        if not isinstance(node,ast.ClassDef) or node.name not in NODES:continue
        schema=next(member for member in node.body if isinstance(member,ast.FunctionDef) and member.name=='define_schema');records=[]
        for call in ast.walk(schema):
            if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute) or call.func.attr!='Input' or not call.args or not isinstance(call.args[0],ast.Constant):continue
            record={'name':call.args[0].value,'type':ast.unparse(call.func.value)}
            for keyword in call.keywords:
                if keyword.arg in {'options','min','max','default'}:
                    try:record[keyword.arg]=ast.literal_eval(keyword.value)
                    except (ValueError,TypeError):pass
            records.append(record)
            for option in record.get('options',[])+([record['min'],record['max']] if 'min' in record else []):
                value=base(node.name)
                if record['name'] in value['settings']:value['settings'][record['name']]=option;await observe(value)
                elif record['name'] in value['settings'].get('should_remesh',{}):value['settings']['should_remesh'][record['name']]=option;await observe(value)
        facts[node.name]=records
    for node in [NODES[0],NODES[2],NODES[3]]:
        for remesh in [{'should_remesh':'false'},{'should_remesh':'false','topology':'quad','target_polycount':100},{'should_remesh':'true','topology':'quad','target_polycount':99},{'should_remesh':'unknown'}]:value=base(node);value['settings']['should_remesh']=remesh;await observe(value)
    for node in [NODES[1],NODES[2],NODES[3],NODES[4],NODES[6]]:
        for batch in [0,1,2,8,9]:
            value=base(node);role='image_style' if node==NODES[6] else 'texture_image';value['images'].append(image(role,batch))
            if node in [NODES[2],NODES[3]]:value['settings']['should_texture']={'should_texture':'true','enable_pbr':True,'texture_prompt':''}
            if node==NODES[6]:value['settings']['text_style_prompt']=''
            await observe(value)
    for node in [NODES[2],NODES[3]]:
        for texture in [{'should_texture':'true','enable_pbr':True,'texture_prompt':' texture '},{'should_texture':'true','texture_prompt':''},{'should_texture':'unknown','enable_pbr':False}]:value=base(node);value['settings']['should_texture']=texture;await observe(value)
    for node,key in [(NODES[0],'prompt'),(NODES[1],'texture_prompt'),(NODES[6],'text_style_prompt')]:
        for text in ['', ' ', '☁'*600, '☁'*601, ' '*600+'x']:value=base(node);value['settings'][key]=text;await observe(value)
    for node in [NODES[1],NODES[2],NODES[3],NODES[6]]:
        value=base(node);role='image_style' if node==NODES[6] else 'texture_image';value['images'].append(image(role));
        if node in [NODES[2],NODES[3]]:value['settings']['should_texture']={'should_texture':'true','enable_pbr':True,'texture_prompt':'x'}
        else:value['settings']['text_style_prompt' if node==NODES[6] else 'texture_prompt']='x'
        await observe(value)
    for batch in [0,1,8,9]:
        value=base(NODES[2]);value['images'][0]['batch']=batch;await observe(value)
        value=base(NODES[3]);value['images']=[image('imageZ',batch),image('imageA',batch)];await observe(value)
    geometry={'__builtins__':__builtins__,'IO':BinaryIO,'Path':Path,'BytesIO':BytesIO,'shutil':shutil}
    selected('sim_api/latest/_util/geometry_types.py',{'File3D'},geometry)
    gltf=json.dumps({'asset':{'version':'2.0'}},separators=(',',':')).encode();gltf+=b' '*((-len(gltf))%4)
    glb=struct.pack('<III',0x46546c67,2,20+len(gltf))+struct.pack('<II',len(gltf),0x4e4f534a)+gltf
    payloads={'glb':glb,'fbx':b'; FBX 7.4.0 project file\n; source File3D opaque byte preservation probe\n\x00\xff'}
    file_probe=[];failure=None
    with tempfile.TemporaryDirectory(prefix='meshy-source-file-') as output:
        file_namespace={'__builtins__':__builtins__,'Path':Path,'BytesIO':BytesIO,'Types':SimpleNamespace(File3D=geometry['File3D']),'get_output_directory':lambda:output}
        async def file_bytes(url,destination,**options):destination.write(payloads[url.rsplit('.',1)[-1]])
        file_namespace['download_url_to_bytesio']=file_bytes
        selected('sim_api_nodes/util/download_helpers.py',{'download_url_to_file_3d'},file_namespace)
        for format in ['glb','fbx']:
            file=await file_namespace['download_url_to_file_3d']('https://output.fixture.invalid/model.'+format,'.'+format.upper(),task_id='task-42')
            file_probe.append({'format':file.format,'disk_backed':file.is_disk_backed,'position_after_creation':file.get_source().tell(),'file_name':'task-42.'+format,'bytes_hex':file.get_bytes().hex(),'saved_bytes_hex':(Path(output)/('task-42.'+format)).read_bytes().hex()})
        (Path(output)/'task-42.fbx').unlink()
        async def failed_bytes(url,destination,**options):
            if url.endswith('.fbx'):raise ValueError('injected second download failure')
            destination.write(glb)
        file_namespace['download_url_to_bytesio']=failed_bytes
        await file_namespace['download_url_to_file_3d']('https://output.fixture.invalid/model.glb','glb',task_id='task-42')
        try:await file_namespace['download_url_to_file_3d']('https://output.fixture.invalid/model.fbx','fbx',task_id='task-42')
        except ValueError:failure={'glb_saved':(Path(output)/'task-42.glb').exists(),'fbx_saved':(Path(output)/'task-42.fbx').exists()}
        else:raise ValueError('missing second-download failure')
    pixels=torch.empty((1,160,160,3),dtype=torch.float32);pixels[...,0]=.5;pixels[...,1]=.25;pixels[...,2]=.125
    actual={'__builtins__':__builtins__,'torch':torch,'np':np,'Image':Image,'BytesIO':BytesIO,'math':math,'uuid':SimpleNamespace(uuid4=lambda:'00000000-0000-4000-8000-000000000000'),'mimetype_to_extension':lambda mime:mime.split('/')[-1].lower()}
    selected('sim_api_nodes/util/conversions.py',{'tensor_to_bytesio','tensor_to_pil','pil_to_bytesio','downscale_image_tensor','_compute_downscale_dims'},actual)
    png=actual['tensor_to_bytesio'](pixels,total_pixels=2048*2048,mime_type=None).getvalue().hex()
    return {'file_probe':file_probe,'file_failure_residue':failure,'actual_png_hex':png,'versions':{'torch':torch.__version__,'pillow':Image.__version__},'sources':{file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in FILES},'cases':cases,'errors':errors,'schema_facts':facts,'scope':'actual source execute/Pydantic validation/order and actual upload helper with shape-only encoder injection; native PNG/resize/file/guest acceptance remains separate'}
parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();result=asyncio.run(generate());encoded=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
if args.check:
    if DEST.read_text()!=encoded:raise SystemExit('source oracle drift')
else:DEST.write_text(encoded)
print(f"Meshy actual source: {len(result['cases'])} success cases, {len(result['errors'])} ordered failures")
