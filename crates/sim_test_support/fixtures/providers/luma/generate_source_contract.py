#!/usr/bin/env python3
import argparse, ast, asyncio, copy, hashlib, inspect, itertools, json
from pathlib import Path
from types import SimpleNamespace

FOLDER=Path(__file__).resolve().parent
SOURCE_ROOT=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI')
SOURCES=['sim_api_nodes/nodes_luma.py','sim_api_nodes/apis/luma.py','sim_api_nodes/util/client.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/upload_helpers.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/_helpers.py','sim_api_nodes/util/download_helpers.py']
class TensorProbe:
    def __init__(self,slot='image',count=1):self.slot=slot;self.shape=(count,720,720,3)
class VideoProbe:
    def __init__(self,duration=4.0,fail=False):self.duration=duration;self.fail=fail
    def get_duration(self):
        if self.fail:raise ValueError('canonical duration unavailable')
        return self.duration

def strip(module):
    for node in ast.walk(module):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            node.returns=None
            for argument in node.args.args+node.args.kwonlyargs:argument.annotation=None
    return ast.fix_missing_locations(module)

def serial(value):
    if isinstance(value,TensorProbe):return {'native_slot':value.slot,'shape':list(value.shape)}
    if isinstance(value,VideoProbe):return {'native_slot':'video','duration':value.duration,'duration_error':value.fail}
    if isinstance(value,dict):return {key:({'ordered_entries':[{'name':name,'image':serial(image)} for name,image in member.items()]} if key=='image_ref' and isinstance(member,dict) else serial(member)) for key,member in value.items()}
    if isinstance(value,(list,tuple)):return [serial(member) for member in value]
    if hasattr(value,'refs'):return {'refs':serial(value.refs)}
    if hasattr(value,'concepts'):return {'concepts':[member.key for member in value.concepts]}
    if hasattr(value,'items'):return {'items':serial(value.items)}
    if hasattr(value,'image'):
        return {'image':serial(value.image),**({'weight':value.weight} if hasattr(value,'weight') else {'mode':value.mode,'value':value.value})}
    return value

async def generate():
    tree=ast.parse((SOURCE_ROOT/SOURCES[0]).read_text());requests=[];media=[];downloads=[];polls=[];contracts={};response_override={}
    namespace={'__builtins__':__builtins__,'__name__':'source_luma','torch':SimpleNamespace(Tensor=TensorProbe),'IO':SimpleNamespace(NodeOutput=lambda *values:values),'ApiEndpoint':lambda path,method='GET':SimpleNamespace(path=path,method=method)}
    api=ast.parse((SOURCE_ROOT/SOURCES[1]).read_text());api.body=[node for node in api.body if not (isinstance(node,ast.Import) and any(alias.name=='torch' for alias in node.names))];exec(compile(api,'<actual-source-api>','exec'),namespace)
    from pydantic import BaseModel
    for member in list(namespace.values()):
        if isinstance(member,type) and member is not BaseModel and issubclass(member,BaseModel):member.model_rebuild(_types_namespace=namespace)
    async def sync(cls,endpoint,data,response_model,**options):
        body=data.model_dump(mode='json',exclude_none=True);requests.append({'path':endpoint.path,'method':endpoint.method,'body':body,'options':options})
        if endpoint.path.startswith('/proxy/luma_2'):
            return response_model.model_validate({'id':'source-id', 'state':'queued',**response_override.get('initial',{})})
        return response_model.model_validate({'id':'source-id','generation_type':'image' if endpoint.path.endswith('/image') else 'video','state':'queued','created_at':'source-date','model':'source-model','request':body,**response_override.get('initial',{})})
    async def poll(cls,poll_endpoint=None,response_model=None,**options):
        polls.append({'path':poll_endpoint.path,'method':poll_endpoint.method,'estimated_duration':options.get('estimated_duration'),'progress_extractor':options.get('progress_extractor') is not None})
        if poll_endpoint.path.startswith('/proxy/luma_2'):
            return response_model.model_validate({'id':'final-id','state':'completed','output':[{'url':'https://media.fixture.invalid/result'}],**response_override.get('final',{})})
        return response_model.model_validate({'id':'final-id','generation_type':'image','state':'completed','created_at':'source-date','model':'source-model','request':{'prompt':'source'},'assets':{'image':'https://media.fixture.invalid/image.png','video':'https://media.fixture.invalid/video.mp4'},**response_override.get('final',{})})
    async def upload_images(cls,image,**options):
        media.append({'operation':'upload_images','image':serial(image),'max_images':options.get('max_images',8),'mime_type':options.get('mime_type'),'total_pixels':4194304,'wait_label':'Uploading','show_batch_index':True});return [f'https://storage.fixture.invalid/{image.slot}-{index}.png' for index in range(min(image.shape[0],options.get('max_images',8)))]
    async def upload_image(cls,image,**options):
        media.append({'operation':'upload_image','image':serial(image),'mime_type':options.get('mime_type'),'total_pixels':4194304,'wait_label':'Uploading'});return f'https://storage.fixture.invalid/{image.slot}-0.png'
    async def upload_video(cls,video,**options):
        media.append({'operation':'upload_video','video':serial(video),'max_duration':options.get('max_duration'),'mime_type':'video/mp4'});return 'https://storage.fixture.invalid/video.mp4'
    async def download_image(url):downloads.append({'kind':'IMAGE','url':url,'color_mode':'RGBA'});return {'canonical_image':url}
    async def download_video(url):downloads.append({'kind':'VIDEO','url':url});return {'canonical_video':url}
    namespace.update(sync_op=sync,poll_op=poll,upload_images_to_simapi=upload_images,upload_image_to_simapi=upload_image,upload_video_to_simapi=upload_video,download_url_to_image_tensor=download_image,download_url_to_video_output=download_video)
    helpers=ast.parse((SOURCE_ROOT/SOURCES[3]).read_text());exec(compile(strip(ast.Module(body=[copy.deepcopy(node) for node in helpers.body if isinstance(node,ast.FunctionDef) and node.name=='validate_string'],type_ignores=[])),'<actual-source-validation>','exec'),namespace)
    selected=[];prices={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            name=ast.unparse(node.targets[0])
            if name in {'LUMA_T2V_AVERAGE_DURATION','LUMA_I2V_AVERAGE_DURATION'}:selected.append(copy.deepcopy(node))
            if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='IO.PriceBadge':prices[name]=ast.literal_eval(next(keyword.value for keyword in node.value.keywords if keyword.arg=='expr'))
        if isinstance(node,ast.AsyncFunctionDef):selected.append(copy.deepcopy(node))
        if isinstance(node,ast.ClassDef) and node.name!='LumaExtension':
            schema=next(call for call in ast.walk(node) if isinstance(call,ast.Call) and ast.unparse(call.func)=='IO.Schema');fields={keyword.arg:keyword.value for keyword in schema.keywords};identifier=ast.literal_eval(fields['node_id']);method=next(member for member in node.body if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)) and member.name=='execute');defaults={argument.arg:ast.literal_eval(value) for argument,value in zip(method.args.args[-len(method.args.defaults):],method.args.defaults)} if method.args.defaults else {}
            contracts[identifier]={'symbol':node.name,'arguments':[argument.arg for argument in method.args.args if argument.arg!='cls'],'defaults':defaults,'schema':ast.unparse(schema),'hidden':ast.unparse(fields.get('hidden',ast.List(elts=[]))),'outputs':ast.unparse(fields['outputs'])}
            if 'price_badge' in fields:
                badge=fields['price_badge'];prices[identifier]=ast.literal_eval(next(keyword.value for keyword in badge.keywords if keyword.arg=='expr')) if isinstance(badge,ast.Call) else ast.unparse(badge)
            members=[copy.deepcopy(member) for member in node.body if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)) and member.name!='define_schema'];selected.append(ast.ClassDef(name=node.name,bases=[],keywords=[],body=members,decorator_list=[]))
    exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<actual-source-execute>','exec'),namespace)
    image=lambda slot='image',count=1:TensorProbe(slot,count)
    ref=lambda count=2:namespace['LumaReferenceChain']()
    def references(count=2):
        chain=ref()
        for index in range(count):chain.add(namespace['LumaReference'](image(f'ref{index}'),0.5))
        return chain
    def concepts():return namespace['LumaConceptChain'](['None','static','static'])
    def keyframes(values=[('fraction',1.0),('seconds',0.0)]):
        chain=namespace['LumaRay32KeyframeChain']()
        for index,(mode,value) in enumerate(values):chain.add(namespace['LumaRay32KeyframeItem'](image(f'key{index}'),mode,value))
        return chain
    common={'prompt':'  source prompt  ','seed':99};bases={
      'LumaConceptsNode':{'concept1':'None','concept2':'static','concept3':'static','concept4':'eye_level','luma_concepts':concepts()},
      'LumaReferenceNode':{'image':image(),'weight':0.555,'luma_ref':references()},
      'LumaImageNode':{**common,'model':'photon-1','aspect_ratio':'16:9','style_image_weight':0.2},
      'LumaImageModifyNode':{**common,'model':'photon-1','image':image(),'image_weight':0.1},
      'LumaVideoNode':{**common,'model':'ray-2','aspect_ratio':'16:9','resolution':'720p','duration':'5s','loop':False},
      'LumaImageToVideoNode':{**common,'model':'ray-2','resolution':'720p','duration':'5s','loop':False,'first_image':image()},
      'LumaImageNode2':{**common,'model':{'model':'uni-1','aspect_ratio':'auto','style':'auto','web_search':False}},
      'LumaImageEditNode2':{**common,'source':image(),'model':{'model':'uni-1','style':'auto','web_search':False}},
      'LumaRay32TextToVideoNode':{**common,'aspect_ratio':'16:9','resolution':'720p','duration':'5s','loop':False},
      'LumaRay32ImageToVideoNode':{**common,'resolution':'720p','loop':False,'start_frame':image()},
      'LumaRay32KeyframeNode':{'image':image(),'position':{'position':'Absolute time (seconds)','seconds':2.5},'keyframes':keyframes()},
      'LumaRay32KeyframesToVideoNode':{**common,'resolution':'720p','duration':'5s','keyframes':keyframes()},
      'LumaRay32VideoEditNode':{**common,'resolution':'720p','strength':'auto','video':VideoProbe()},
      'LumaRay32VideoReframeNode':{**common,'resolution':'720p','aspect_ratio':'16:9','video':VideoProbe(10)},
      'LumaRay32ExtendVideoNode':{**common,'resolution':'720p','source_generation_id':'  old-id  ','direction':{'direction':'Forward (continue after)','loop':False}}}
    cases=[];errors=[];response_errors=[]
    async def observe(node,arguments,failure=False,response=None):
        requests.clear();media.clear();downloads.clear();polls.clear();response_override.clear();response_override.update(response or {})
        try:
            result=namespace[contracts[node]['symbol']].execute(**copy.deepcopy(arguments))
            if inspect.isawaitable(result):result=await result
        except Exception as error:
            if not failure:raise
            destination=response_errors if response else errors;destination.append({'input':{'node':node,'arguments':serial(arguments)},'message':str(error),'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),'polls':copy.deepcopy(polls),'downloads':copy.deepcopy(downloads),**({'response_override':response} if response else {})})
        else:
            if failure:raise ValueError(f'Expected source failure {node}')
            encoded=serial(arguments)
            cases.append({'input':{'node':node,'arguments':encoded},'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),'polls':copy.deepcopy(polls),'downloads':copy.deepcopy(downloads),'outputs':serial(result)})
    for node,base in bases.items():await observe(node,base)
    for concept in namespace['get_luma_concepts'](True):await observe('LumaConceptsNode',{**bases['LumaConceptsNode'],'concept1':concept,'luma_concepts':None})
    for weight in [0,1,.125,.135,.145,.555,2.675,-.125]:await observe('LumaReferenceNode',{**bases['LumaReferenceNode'],'weight':weight})
    await observe('LumaRay32KeyframeNode',{**bases['LumaRay32KeyframeNode'],'position':{'position':'Fraction of duration (0.0-1.0)','fraction':'0.25'}})
    for model,ratio in itertools.product(['photon-1','photon-flash-1'],['1:1','16:9','9:16','4:3','3:4','21:9','9:21']):await observe('LumaImageNode',{**bases['LumaImageNode'],'model':model,'aspect_ratio':ratio,'image_luma_ref':references(6),'style_image':image('style'),'character_image':image('character',5)})
    for index in [0,1]:
        chain=references(6);chain.refs[index].weight=1.5
        await observe('LumaImageNode',{**bases['LumaImageNode'],'image_luma_ref':chain},True)
    await observe('LumaImageNode',{**bases['LumaImageNode'],'style_image':image('style'),'style_image_weight':1.5},True)
    for weight in [-1,0,.1,.98,1,2,.445]:await observe('LumaImageModifyNode',{**bases['LumaImageModifyNode'],'image_weight':weight,'prompt':''})
    for node in ['LumaVideoNode','LumaImageToVideoNode']:
        for model,resolution,duration in itertools.product(['ray-2','ray-flash-2','ray-1-6'],['540p','720p','1080p','4k'],['5s','9s']):await observe(node,{**bases[node],'model':model,'resolution':resolution,'duration':duration,'loop':True,'luma_concepts':concepts()})
    await observe('LumaImageToVideoNode',{**bases['LumaImageToVideoNode'],'first_image':None,'last_image':image('last'),'prompt':''})
    await observe('LumaImageToVideoNode',{**bases['LumaImageToVideoNode'],'first_image':image('first'),'last_image':image('last')})
    for node,max_refs in [('LumaImageNode2',9),('LumaImageEditNode2',8)]:
        for model,style in itertools.product(['uni-1','uni-1-max'],['auto','manga']):
            dynamic={**bases[node]['model'],'model':model,'style':style,'image_ref':{'image_9':image('ninth'),'image_1':image('first')}}
            if node=='LumaImageNode2':dynamic['aspect_ratio']='2:3'
            await observe(node,{**bases[node],'model':dynamic})
        dynamic={**bases[node]['model'],'image_ref':{f'image_{index}':image(f'ref{index}') for index in range(max_refs+1)}}
        await observe(node,{**bases[node],'model':dynamic},True)
    for resolution,duration,loop in itertools.product(['360p','540p','720p','1080p'],['5s','10s'],[False,True]):await observe('LumaRay32TextToVideoNode',{**bases['LumaRay32TextToVideoNode'],'resolution':resolution,'duration':duration,'loop':loop},loop and duration=='10s')
    for node in ['LumaRay32ImageToVideoNode','LumaRay32KeyframesToVideoNode','LumaRay32ExtendVideoNode']:
        for resolution in ['360p','540p','720p','1080p']:await observe(node,{**bases[node],'resolution':resolution})
    for duration,strength in itertools.product([4,5,5.01,18],['auto','adhere_1','adhere_2','adhere_3','flex_1','flex_2','flex_3','reimagine_1','reimagine_2','reimagine_3']):await observe('LumaRay32VideoEditNode',{**bases['LumaRay32VideoEditNode'],'video':VideoProbe(duration),'strength':strength})
    await observe('LumaRay32VideoEditNode',{**bases['LumaRay32VideoEditNode'],'video':VideoProbe(fail=True)})
    for ratio,resolution in itertools.product(['16:9','9:16','1:1','4:3','3:4','21:9'],['360p','540p','720p','1080p']):await observe('LumaRay32VideoReframeNode',{**bases['LumaRay32VideoReframeNode'],'aspect_ratio':ratio,'resolution':resolution},resolution=='1080p' and ratio in ['9:16','3:4'])
    await observe('LumaRay32ExtendVideoNode',{**bases['LumaRay32ExtendVideoNode'],'direction':{'direction':'Backward (lead-in before)'}})
    await observe('LumaRay32ExtendVideoNode',{**bases['LumaRay32ExtendVideoNode'],'direction':{'direction':'Forward (continue after)','loop':True}})
    await observe('LumaRay32ImageToVideoNode',{**bases['LumaRay32ImageToVideoNode'],'start_frame':image('start'),'end_frame':image('end')})
    await observe('LumaRay32ImageToVideoNode',{**bases['LumaRay32ImageToVideoNode'],'start_frame':None,'end_frame':image('end')})
    for values,duration,failure in [([('fraction',.004166666666666667),('fraction',1)],'5s',False),([('fraction',.0125),('fraction',1)],'5s',False),([('fraction',-1),('fraction',2)],'5s',False),([('seconds',5.01),('fraction',0)],'5s',True),([('seconds',2.5),('fraction',.5)],'5s',True),([],'5s',True),([('fraction',0)],'5s',True),([('fraction',index/64) for index in range(65)],'10s',True),([('fraction',1),('seconds',0)],'10s',False)]:await observe('LumaRay32KeyframesToVideoNode',{**bases['LumaRay32KeyframesToVideoNode'],'keyframes':keyframes(values),'duration':duration},failure)
    for node in bases:
        if 'prompt' not in bases[node]:continue
        for prompt in ['', ' ', 'x'*6001]:
            failure=(node not in {'LumaImageModifyNode','LumaImageToVideoNode'} and (len(prompt.strip())< (3 if node in {'LumaImageNode','LumaVideoNode'} else 1) or len(prompt)>6000))
            if node in {'LumaRay32VideoReframeNode','LumaRay32ExtendVideoNode'}:failure=len(prompt)<1 or len(prompt)>6000
            if node in {'LumaImageNode','LumaVideoNode'} and len(prompt)>6000:failure=False
            await observe(node,{**bases[node],'prompt':prompt},failure)
    await observe('LumaImageToVideoNode',{**bases['LumaImageToVideoNode'],'first_image':None},True)
    await observe('LumaRay32ImageToVideoNode',{**bases['LumaRay32ImageToVideoNode'],'start_frame':None},True)
    await observe('LumaRay32ImageToVideoNode',{**bases['LumaRay32ImageToVideoNode'],'end_frame':image(),'loop':True},True)
    await observe('LumaRay32ExtendVideoNode',{**bases['LumaRay32ExtendVideoNode'],'source_generation_id':' '},True)
    await observe('LumaImageNode2',{**bases['LumaImageNode2'],'model':{**bases['LumaImageNode2']['model'],'style':'manga','aspect_ratio':'16:9'}},True)
    for node in ['LumaImageNode2','LumaRay32TextToVideoNode']:
        for response in [{'initial':{'id':None}},{'initial':{'id':''}},{'final':{'output':[]}},{'final':{'output':None,'failure_reason':'blocked','failure_code':'source-code'}},{'final':{'output':[{'url':''}]}}]:await observe(node,bases[node],True,response)
    polling={}
    client=ast.parse((SOURCE_ROOT/SOURCES[2]).read_text())
    for node in client.body:
        if isinstance(node,ast.Assign) and any(ast.unparse(target) in {'COMPLETED_STATUSES','FAILED_STATUSES','QUEUED_STATUSES'} for target in node.targets):polling[ast.unparse(node.targets[0])]=sorted(str(value).lower() for value in ast.literal_eval(node.value))
    return {'schema_version':1,'namespace':'zed.sim.provider.luma','scope':'Pinned actual AST/Pydantic request and custom-chain source oracle. Native uploads, chain registry, download allocation, clock and price engine are canonical owner gates; structural probes claim no native execution. Modern refs use ordered_entries at the native owner boundary to preserve Python insertion order.','source_sha256':{path:hashlib.sha256((SOURCE_ROOT/path).read_bytes()).hexdigest() for path in SOURCES},'contracts':contracts,'cases':cases,'preflight_errors':errors,'response_errors':response_errors,'price_expressions':prices,'polling':polling,'api_schemas':{name:namespace[name].model_json_schema() for name in ['LumaImageGenerationRequest','LumaGenerationRequest','Luma2GenerationRequest','LumaGeneration','Luma2Generation']}}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT);parser.add_argument('--check',action='store_true');arguments=parser.parse_args();SOURCE_ROOT=arguments.source_root
    encoded=(json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n').encode();destination=FOLDER/'source-contract.json'
    if arguments.check:
        if destination.read_bytes()!=encoded:raise SystemExit('Luma source oracle differs')
    else:destination.write_bytes(encoded)
