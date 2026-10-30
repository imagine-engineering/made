"""Pinned Hunyuan3D execute/API AST oracle; media owner boundaries are explicit."""
import argparse, ast, asyncio, copy, hashlib, json, zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image

torch.set_num_threads(1)
ROOT = Path('/home/asvegah/repos/sim/projects/sim/ComfyUI')
DESTINATION = Path(__file__).with_name('source-contract.json')
SOURCES = ['sim_api_nodes/nodes_hunyuan3d.py','sim_api_nodes/apis/hunyuan3d.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/client.py','sim_api_nodes/util/upload_helpers.py','sim_api_nodes/util/download_helpers.py']
NODES = ['Tencent3DPartNode','Tencent3DTextureEditNode','TencentImageToModelNode','TencentModelTo3DUVNode','TencentSmartTopologyNode','TencentTextToModelNode']

def selected(path, names, namespace):
    nodes = [copy.deepcopy(node) for node in ast.parse((ROOT/path).read_text()).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.name in names]
    for node in nodes:
        for member in ast.walk(node):
            if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
                member.returns=None
                for argument in [*member.args.args,*member.args.posonlyargs,*member.args.kwonlyargs]: argument.annotation=None
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(ROOT/path),'exec'),namespace)

class ImageProbe:
    def __init__(self,role,width=256,height=128): self.role,self.shape=role,(1,height,width,4)
    def movedim(self,source,destination):
        result=ImageProbe(self.role)
        shape=list(self.shape);shape.insert(destination % len(shape),shape.pop(source));result.shape=tuple(shape);return result

class FileProbe:
    def __init__(self,format='FBX',source=None,file_format=None):
        self.format=file_format or format
        self.bytes=source.getvalue() if source is not None else b'actual source model bytes'
    def get_data(self):return self.bytes

def json_output(value):
    if value is None or isinstance(value,str):return value
    if isinstance(value,FileProbe):return {'file_format':value.format,'bytes_hex':value.bytes.hex()}
    if isinstance(value,torch.Tensor):return {'shape':list(value.shape),'pixels':value.flatten().tolist()}
    return value

async def generate():
    requests=[];uploads=[];polls=[];downloads=[];resizes=[]
    namespace={'__builtins__':__builtins__,'IO':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *values:values),'Types':SimpleNamespace(File3D=FileProbe),'BytesIO':BytesIO,'zipfile':zipfile,'torch':torch,'np':np,'Image':Image}
    exec(compile((ROOT/SOURCES[1]).read_text(),str(ROOT/SOURCES[1]),'exec'),namespace)
    selected(SOURCES[2],{'get_image_dimensions','validate_image_dimensions','validate_string'},namespace)
    def upscale(image,width,height,method,crop):
        resizes.append({'role':image.role,'width':width,'height':height,'method':method,'crop':crop})
        result=ImageProbe(image.role,width,height);result.shape=(1,4,height,width);return result
    namespace['common_upscale']=upscale
    selected(SOURCES[3],{'downscale_image_tensor_by_max_side','bytesio_to_image_tensor'},namespace)
    selected(SOURCES[0],{'_is_tencent_rate_limited','ObjZipResult','download_and_extract_obj_zip','get_file_from_response'},namespace)
    async def upload_image(cls,image,**options):
        uploads.append({'role':image.role,'width':image.shape[2],'height':image.shape[1],**options});return 'https://storage.fixture.invalid/'+image.role
    async def upload_model(cls,model,format):
        uploads.append({'role':'model_3d','format':format,'bytes_hex':model.get_data().hex()});return 'https://storage.fixture.invalid/model_3d'
    response_files=[];obj_mode='zip'
    async def sync(cls,endpoint,data,response_model,**options):
        requests.append({'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(exclude_none=True),'tencent_rate_limit':options.get('is_rate_limited') is namespace['_is_tencent_rate_limited']})
        return response_model.model_validate({'Response':{'JobId':'source-task-42'}})
    async def poll(cls,endpoint,data,response_model,**options):
        polls.append({'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(exclude_none=True)})
        return response_model.model_validate({'Response':{'Status':'Done','ResultFile3Ds':response_files}})
    async def file(url,format,**options):
        downloads.append({'url':url,'kind':'file','format':format,**options});return {'download':url,'format':format,**options}
    async def image(url):downloads.append({'url':url,'kind':'image','mode':'RGBA'});return {'download':url,'mode':'RGBA'}
    async def obj(url):
        downloads.append({'url':url,'kind':'obj_archive','mode':'RGB'})
        return namespace['ObjZipResult'](FileProbe(file_format='obj'),texture='source RGB texture' if obj_mode=='zip' else None)
    namespace.update(ApiEndpoint=lambda path,method='GET':SimpleNamespace(path=path,method=method),sync_op=sync,poll_op=poll,upload_image_to_simapi=upload_image,upload_3d_model_to_simapi=upload_model,download_url_to_file_3d=file,download_url_to_image_tensor=image,download_and_extract_obj_zip=obj)
    node_classes=[]
    for node in ast.parse((ROOT/SOURCES[0]).read_text()).body:
        if isinstance(node,ast.ClassDef) and node.name in NODES:
            node=copy.deepcopy(node);node.body=[member for member in node.body if isinstance(member,ast.Assign) or isinstance(member,ast.AsyncFunctionDef) and member.name=='execute'];node_classes.append(node)
    for node in node_classes:
        for member in ast.walk(node):
            if isinstance(member,ast.AsyncFunctionDef):
                member.returns=None
                for arg in member.args.args:arg.annotation=None
    exec(compile(ast.fix_missing_locations(ast.Module(body=node_classes,type_ignores=[])),str(ROOT/SOURCES[0]),'exec'),namespace)
    def candidate(node,**extra):return {'node':node,'model':'3.0','prompt':'  cloud ☁  ','face_count':500000,'generate_type':{'generate_type':'Normal','pbr':False},'seed':42,'model_format':'FBX','polygon_type':'triangle','face_level':'medium','images':{},**extra}
    def kwargs(value):
        result={'seed':value['seed']};node=value['node']
        if node.endswith('TextToModelNode') or node.endswith('ImageToModelNode'):
            result.update(model=value['model'],face_count=value['face_count'],generate_type=value['generate_type'])
            if node.endswith('TextToModelNode'):result['prompt']=value['prompt']
            else:
                for role,dimensions in value['images'].items():result[role]=ImageProbe(role,**dimensions)
        else:
            result['model_3d']=FileProbe(value['model_format'])
            if node=='Tencent3DTextureEditNode':result['prompt']=value['prompt']
            if node=='TencentSmartTopologyNode':result.update(polygon_type=value['polygon_type'],face_level=value['face_level'])
        return result
    def clear():
        for values in [requests,uploads,polls,downloads,resizes]:values.clear()
    files=[{'Type':kind.upper(),'Url':'https://download.fixture.invalid/'+kind} for kind in ['obj','glb','fbx','uv_image','texture_image']]
    cases=[];errors=[]
    async def observe(value,available=files,mode='zip'):
        nonlocal response_files,obj_mode
        clear();response_files=available;obj_mode=mode
        result=await namespace[value['node']].execute(**kwargs(value))
        cases.append({'input':value,'files':available,'obj_mode':mode,'request':copy.deepcopy(requests[0]),'uploads':copy.deepcopy(uploads),'resizes':copy.deepcopy(resizes),'poll':copy.deepcopy(polls[0]),'downloads':copy.deepcopy(downloads),'outputs':[json_output(item) for item in result]})
    for node in NODES:
        extras={'images':{'image':{'width':256,'height':128}}} if node=='TencentImageToModelNode' else {'model_format':'OBJ'} if node=='TencentSmartTopologyNode' else {}
        await observe(candidate(node,**extras))
    for model,generate in [('3.1',{'generate_type':'Geometry'}),('3.0',{'generate_type':'LowPoly','polygon_type':'quadrilateral','pbr':True})]:
        await observe(candidate('TencentTextToModelNode',model=model,generate_type=generate,face_count=3000,prompt='☁'*1024))
    await observe(candidate('TencentImageToModelNode',images={role:{'width':9800 if role=='image_right' else 256,'height':257 if role=='image_right' else 128} for role in ['image','image_left','image_right','image_back']},generate_type={'generate_type':'Normal','pbr':True}))
    await observe(candidate('TencentImageToModelNode',images={'image':{'width':128,'height':128}}),[item for item in files if item['Type']!='OBJ'])
    await observe(candidate('TencentTextToModelNode'),[item for item in files if item['Type']!='OBJ'])
    await observe(candidate('TencentImageToModelNode',images={'image':{'width':128,'height':128}}),mode='raw')
    await observe(candidate('TencentModelTo3DUVNode',model_format='GLB'),[item for item in files if item['Type']!='UV_IMAGE'])
    for polygon in ['triangle','quadrilateral']:
        for level in ['medium','high','low']:await observe(candidate('TencentSmartTopologyNode',model_format='GLB',polygon_type=polygon,face_level=level))
    invalid=[candidate('TencentTextToModelNode',prompt=' \u001c '),candidate('TencentTextToModelNode',prompt='☁'*1025),candidate('TencentTextToModelNode',model='3.1',generate_type={'generate_type':'LowPoly'}),candidate('TencentImageToModelNode',images={'image':{'width':127,'height':128}}),candidate('Tencent3DTextureEditNode',model_format='OBJ'),candidate('Tencent3DPartNode',model_format='GLB'),candidate('TencentModelTo3DUVNode',model_format='STL'),candidate('TencentSmartTopologyNode',model_format='FBX')]
    for value in invalid:
        clear()
        try:await namespace[value['node']].execute(**kwargs(value))
        except Exception as error:errors.append({'input':value,'error':str(error),'requests':len(requests),'uploads':len(uploads)})
        else:raise ValueError('expected source preflight error')
    rate=[{'status':status,'body':body,'rate_limited':namespace['_is_tencent_rate_limited'](status,body)} for status,body in [(400,{'Response':{'Error':{'Code':'xxRequestLimitExceeded.zz'}}}),(400,{'Error':{'Code':'RequestLimitExceeded'}}),(429,{'Response':{'Error':{'Code':'RequestLimitExceeded'}}}),(400,{})]]
    selected(SOURCES[0],{'download_and_extract_obj_zip'},namespace)
    directory=DESTINATION.parent/'archives';directory.mkdir(exist_ok=True)
    pixel=BytesIO();Image.new('RGBA',(1,1),(128,64,32,17)).save(pixel,format='PNG')
    entries=[('first.OBJ',b'first mesh'),('ignored.mtl',b'ignored material'),('old.png',pixel.getvalue()),('folder/a_METALLIC.PNG',pixel.getvalue()),('folder/a_NORMAL.PNG',pixel.getvalue()),('folder/a_ROUGHNESS.PNG',pixel.getvalue()),('new.PNG',pixel.getvalue()),('last.obj',b'last mesh')]
    archive_cases=[]
    for name,contents,is_zip in [('ordered.zip',entries,True),('no-obj.zip',[('texture.png',pixel.getvalue())],True),('invalid-overwritten.zip',[('mesh.obj',b'mesh'),('old.png',b'invalid image'),('new.png',pixel.getvalue())],True),('raw.obj',[],False)]:
        encoded=BytesIO()
        if is_zip:
            with zipfile.ZipFile(encoded,'w',compression=zipfile.ZIP_STORED) as archive:
                for entry,bytes in contents:
                    info=zipfile.ZipInfo(entry,date_time=(2020,1,1,0,0,0));archive.writestr(info,bytes)
        else:encoded.write(b'raw actual OBJ bytes')
        data=encoded.getvalue();path=directory/name
        if not args.check:path.write_bytes(data)
        elif path.read_bytes()!=data:raise ValueError('Archive fixture changed')
        async def download_bytes(url,sink):sink.write(data)
        namespace['download_url_to_bytesio']=download_bytes
        try:
            result=await namespace['download_and_extract_obj_zip']('https://download.fixture.invalid/'+name)
            observed={field:json_output(getattr(result,field)) for field in ['obj','texture','metallic','normal','roughness']}
        except Exception as error:observed={'error_type':type(error).__name__,'error':str(error).split('<_io.BytesIO')[0]}
        archive_cases.append({'file':'archives/'+name,'sha256':hashlib.sha256(data).hexdigest(),'entries':[{'name':entry,'bytes_hex':bytes.hex()} for entry,bytes in contents],'result':observed})
    schema_facts={}
    for node in ast.parse((ROOT/SOURCES[0]).read_text()).body:
        if isinstance(node,ast.ClassDef) and node.name in NODES:
            schema=next(member for member in node.body if isinstance(member,ast.FunctionDef) and member.name=='define_schema')
            facts=[]
            for call in ast.walk(schema):
                if isinstance(call,ast.Call):
                    literals={}
                    for keyword in call.keywords:
                        try:literals[keyword.arg]=ast.literal_eval(keyword.value)
                        except (ValueError,TypeError):pass
                    if literals:facts.append({'call':ast.unparse(call.func),'literals':literals})
            schema_facts[node.name]=facts
    client_defaults={}
    for path,names in [(SOURCES[4],['sync_op_raw','poll_op_raw']),(SOURCES[6],['download_url_to_bytesio'])]:
        for node in ast.parse((ROOT/path).read_text()).body:
            if isinstance(node,ast.AsyncFunctionDef) and node.name in names:
                values={}
                for argument,default in zip(node.args.kwonlyargs,node.args.kw_defaults):
                    if default is not None:
                        try:values[argument.arg]=ast.literal_eval(default)
                        except (ValueError,TypeError):pass
                client_defaults[node.name]=values
    return {'schema_version':1,'namespace':'zed.sim.provider.hunyuan3d','scope':'Actual pinned execute/API/validation/downscale AST; upload encoding, native File3D ownership, RGB archive extraction and allocation remain injected canonical boundaries.','source_sha256':{path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in SOURCES},'cases':cases,'preflight_errors':errors,'rate_limit_cases':rate,'archive_cases':archive_cases,'schema_facts':schema_facts,'client_defaults':client_defaults}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--source-root',type=Path,default=ROOT);args=parser.parse_args();ROOT=args.source_root
    encoded=(json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n').encode()
    if args.check:
        if DESTINATION.read_bytes()!=encoded:raise SystemExit('Hunyuan3D source contract differs')
    else:DESTINATION.write_bytes(encoded)
