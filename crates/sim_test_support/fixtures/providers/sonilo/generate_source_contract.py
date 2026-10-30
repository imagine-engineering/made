import argparse,ast,asyncio,base64,copy,hashlib,io,json,logging,re,time,wave
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urljoin
import aiohttp
SOURCE=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI');ROOT=Path(__file__).resolve().parent
WAV=(ROOT/'source-response.wav').read_bytes()
MP4=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI-Frontend/browser_tests/assets/plain_video.mp4').read_bytes()
def strip(node):
 for member in ast.walk(node):
  if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
   member.returns=None
   for argument in member.args.args+member.args.kwonlyargs:argument.annotation=None
 return ast.fix_missing_locations(node)
class VideoProbe:
 def __init__(self,duration=30.0,fail=False):self.duration=duration;self.fail=fail
 def get_duration(self):
  if self.fail:raise ValueError('source video duration unavailable')
  return self.duration
 def save_to(self,target,**options):target.write(MP4)
def serial(value):
 if isinstance(value,VideoProbe):return {'native_slot':'video','duration':value.duration,'fail_duration':value.fail}
 if isinstance(value,dict):return {key:serial(member) for key,member in value.items()}
 if isinstance(value,(tuple,list)):return [serial(member) for member in value]
 if isinstance(value,bytes):return {'bytes_hex':value.hex()}
 return value
async def generate():
 logging.disable(logging.CRITICAL);requests=[];operations=[];progress=[];times=[];interrupt=False;read_count=0;script=[];http_status=200;http_json={};http_text='error';json_failure=False
 hidden=SimpleNamespace(auth_token_sim_org='capability-token',api_key_sim_org='capability-api-key',sim_usage_source=None)
 class NodeOutput:
  def __init__(self,*values):self.values=values
 namespace={'__builtins__':__builtins__,'IO':SimpleNamespace(ComfyNode=type('ComfyNode',(),{'hidden':hidden}),NodeOutput=NodeOutput),'aiohttp':aiohttp,'json':json,'base64':base64,'logging':logging,'logger':logging.getLogger('sonilo-source-probe'),'urljoin':urljoin,'BytesIO':io.BytesIO,'get_deploy_environment':lambda:'host','time':SimpleNamespace(monotonic=lambda:times.pop(0) if times else 0.0),'get_node_id':lambda cls:'source-node','is_processing_interrupted':lambda:interrupt,'ApiEndpoint':lambda path,method='GET',**options:SimpleNamespace(path=path,method=method,headers={})}
 class Interrupted(Exception):pass
 namespace['ProcessingInterrupted']=Interrupted
 helpers=ast.parse((SOURCE/'sim_api_nodes/util/_helpers.py').read_text());selected=[copy.deepcopy(node) for node in helpers.body if isinstance(node,ast.FunctionDef) and node.name in ['get_auth_header','get_usage_source','get_sim_api_headers','default_base_url']];namespace['args']=SimpleNamespace(sim_api_base='https://fixture.invalid/api');exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<actual-source-auth-helpers>','exec'),namespace)
 validation=ast.parse((SOURCE/'sim_api_nodes/util/validation_utils.py').read_text());exec(compile(strip(ast.Module(body=[copy.deepcopy(node) for node in validation.body if isinstance(node,ast.FunctionDef) and node.name=='validate_string'],type_ignores=[])),'<actual-source-prompt-validation>','exec'),namespace)
 upload=ast.parse((SOURCE/'sim_api_nodes/util/upload_helpers.py').read_text());method=copy.deepcopy(next(node for node in upload.body if isinstance(node,ast.AsyncFunctionDef) and node.name=='upload_video_to_simapi'))
 namespace.update(Types=SimpleNamespace(VideoContainer=SimpleNamespace(MP4=SimpleNamespace(value='mp4')),VideoCodec=SimpleNamespace(H264='H264')),uuid=SimpleNamespace(uuid4=lambda:'source-video-uuid'))
 async def upload_file(cls,data,name,mime,label):operations.append({'operation':'canonical_upload_file_boundary','bytes':len(data.getvalue()),'sha256':hashlib.sha256(data.getvalue()).hexdigest(),'filename':name,'mime_type':mime,'wait_label':label});return 'https://fixture.invalid/uploaded.mp4'
 namespace['upload_file_to_simapi']=upload_file;exec(compile(strip(ast.Module(body=[method],type_ignores=[])),'<actual-source-video-upload-preparation>','exec'),namespace)
 original_upload=namespace['upload_video_to_simapi']
 async def upload_video(cls,video,**options):operations.append({'operation':'upload_video_to_simapi','video':serial(video),'options':options});return await original_upload(cls,video,**options)
 namespace['upload_video_to_simapi']=upload_video
 def audio_owner(data):operations.append({'operation':'canonical_audio_decode_boundary','bytes_hex':data.hex()});return {'encoded_source_probe_only':data.hex()}
 namespace['audio_bytes_to_audio_input']=audio_owner
 class Response:
  def __init__(self):self.status=http_status;self.content=self
  async def readline(self):
   nonlocal read_count
   read_count+=1;return script.pop(0) if script else b''
  async def json(self):
   if json_failure:raise ValueError('source error JSON unavailable')
   return http_json
  async def text(self):return http_text
  async def __aenter__(self):return self
  async def __aexit__(self,*args):return False
 class Session:
  def __init__(self,timeout):self.timeout=timeout
  async def __aenter__(self):return self
  async def __aexit__(self,*args):return False
  def post(self,url,data,headers):
   fields=[[options['name'],value] for options,headers,value in data._fields];payload=data();requests.append({'fields':fields,'url':url,'method':'POST','headers':headers,'content_type':payload.content_type,'body_hex':payload._value.hex(),'timeout':{'total':self.timeout.total,'sock_read':self.timeout.sock_read}});return Response()
 namespace['aiohttp']=SimpleNamespace(FormData=aiohttp.FormData,ClientTimeout=aiohttp.ClientTimeout,ClientSession=Session)
 namespace['PromptServer']=SimpleNamespace(instance=SimpleNamespace(send_progress_text=lambda message,node_id:progress.append(message)))
 tree=ast.parse((SOURCE/'sim_api_nodes/nodes_sonilo.py').read_text());contracts={};nodes=[]
 for node in tree.body:
  if isinstance(node,ast.ClassDef) and node.name in ['SoniloVideoToMusic','SoniloTextToMusic']:
   method=copy.deepcopy(next(member for member in node.body if isinstance(member,ast.AsyncFunctionDef) and member.name=='execute'));schema=next(call for call in ast.walk(node) if isinstance(call,ast.Call) and ast.unparse(call.func)=='IO.Schema');fields={keyword.arg:keyword.value for keyword in schema.keywords};defaults={arg.arg:ast.literal_eval(default) for arg,default in zip(method.args.args[-len(method.args.defaults):],method.args.defaults)};contracts[node.name]={'arguments':[arg.arg for arg in method.args.args if arg.arg!='cls'],'defaults':defaults,'schema':ast.unparse(schema),'hidden':ast.unparse(fields['hidden']),'outputs':ast.unparse(fields['outputs']),'price_badge':ast.unparse(fields['price_badge'])};nodes.append(ast.ClassDef(name=node.name,bases=[ast.Attribute(value=ast.Name(id='IO',ctx=ast.Load()),attr='ComfyNode',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[]))
  elif isinstance(node,ast.AsyncFunctionDef) and node.name in ['_stream_sonilo_music','_extract_error_message']:nodes.append(copy.deepcopy(node))
 exec(compile(strip(ast.Module(body=nodes,type_ignores=[])),'<actual-source-Sonilo-nodes-and-stream>','exec'),namespace)
 cases=[];errors=[]
 def event(value):return json.dumps(value,separators=(',',':')).encode()+b'\n'
 base_lines=[event({'type':'duration','duration_sec':2.5}),event({'type':'title','title':'Source title'}),event({'type':'audio_chunk','data':base64.b64encode(WAV[:20]).decode()}),event({'type':'audio_chunk','data':base64.b64encode(WAV[20:]).decode()}),event({'type':'complete'})]
 async def observe(node,arguments,lines=None,status=200,error_json=None,error_text='error',fail_json=False,failure=False,clock=None,cancel=False):
  nonlocal script,http_status,http_json,http_text,json_failure,times,interrupt,read_count
  requests.clear();operations.clear();progress.clear();read_count=0;script=copy.deepcopy(base_lines if lines is None else lines);original_lines=list(script);http_status=status;http_json={} if error_json is None else error_json;http_text=error_text;json_failure=fail_json;times=list(clock or [0.0,1.0,2.0]);interrupt=cancel
  record={'input':{'node':node,'arguments':serial(arguments)},'lines_hex':[line.hex() for line in original_lines],'status':status,'error_json':http_json,'error_text':http_text,'fail_json':fail_json,'clock':clock or [0.0,1.0,2.0],'cancel':cancel}
  try:result=await namespace[node].execute(**arguments)
  except Exception as error:
   if not failure:raise
   record.update(error_type=type(error).__name__,message=re.sub(r'0x[0-9a-fA-F]+','<address>',str(error)));errors.append(record)
  else:
   if failure:raise ValueError('Expected source failure '+node)
   record['outputs']=result.values;cases.append(record)
  record.update(requests=copy.deepcopy(requests),operations=copy.deepcopy(operations),progress=copy.deepcopy(progress),read_count=read_count)
 for prompt in [' source prompt ','music ~ + & = / ? 🌊','\u001c music \u001f','x'*1000]:
  for duration in [1,30,360,0,-1,361]:await observe('SoniloTextToMusic',{'prompt':prompt,'duration':duration,'seed':0xFFFFFFFFFFFFFFFF})
 for prompt in ['', ' \t\n',' source video ','x'*1500]:
  for duration in [0.0,30.0,360.0]:await observe('SoniloVideoToMusic',{'video':VideoProbe(duration),'prompt':prompt,'seed':-1})
 for prompt in ['', ' \t\n','\u001c\u001f','x'*1001]:await observe('SoniloTextToMusic',{'prompt':prompt},failure=True)
 for video in [VideoProbe(360.0001),VideoProbe(30,True)]:await observe('SoniloVideoToMusic',{'video':video},failure=True)
 variants=[
  [b'\n',b'not-json\n',event({'type':'ignored'}),*base_lines],
  [event({'type':'titles','titles':['one','two']}),event({'type':'title','title':''}),*base_lines],
  [event({'type':'audio_chunk','stream_index':2,'data':base64.b64encode(WAV).decode()}),event({'type':'audio_chunk','stream_index':1,'data':base64.b64encode(WAV).decode()})],
  [event({'type':'audio_chunk','stream_index':-1,'data':base64.b64encode(b'other').decode()}),event({'type':'audio_chunk','stream_index':0,'data':base64.b64encode(WAV).decode()})],
  [event({'type':'audio_chunk','data':base64.b64encode(WAV).decode()+'$! \t'}),event({'type':'complete'}),event({'type':'error','message':'late ignored'})],
  [event({'type':'audio_chunk','data':''}),event({'type':'complete'})],
 ]
 for lines in variants:await observe('SoniloTextToMusic',{'prompt':'music'},lines=lines)
 failures=[[],[event({'type':'complete'})],[event({'type':'error'})],[event({'type':'error','code':'SOURCE','message':'failed'})],[b'\xff\n'],[event({'type':'audio_chunk'})],[event({'type':'audio_chunk','data':'a'})],[event({'type':'duration','duration_sec':'bad'})],[b'[]\n'],[event({'type':'audio_chunk','stream_index':[],'data':''})]]
 for lines in failures:await observe('SoniloTextToMusic',{'prompt':'music'},lines=lines,failure=True)
 await observe('SoniloTextToMusic',{'prompt':'music'},cancel=True,failure=True)
 for status,body in [(400,{}),(429,{'detail':{'message':'quota'}}),(500,{'detail':'server'}),(403,['unsupported JSON'])]:await observe('SoniloTextToMusic',{'prompt':'music'},status=status,error_json=body,error_text='fallback',failure=True)
 await observe('SoniloTextToMusic',{'prompt':'music'},status=502,fail_json=True,error_text='gateway',failure=True)
 await observe('SoniloTextToMusic',{'prompt':'music'},clock=[10.0,10.1,10.5])
 wire_probes=[]
 for fields in [[('field','~ + /?&= 🌊')],[('same','first'),('same','second'),('empty','')],[('unicode','é漢字'),('spaces',' a b ')]]:
  form=aiohttp.FormData()
  for name,value in fields:form.add_field(name,value)
  payload=form();wire_probes.append({'fields':fields,'content_type':payload.content_type,'body_hex':payload._value.hex(),'classification':'minimum-supported aiohttp wire probe, not a source node branch'})
 scalar_probes=[];base64_probes=[]
 def scalar(value,style):
  record={'value':value,'style':style}
  try:record['output']=str(value) if style=='PythonString' else f'{value:.1f}' if style=='FixedOneDecimal' else str(int(value))
  except Exception as error:record['error_type']=type(error).__name__
  if record not in scalar_probes:scalar_probes.append(record)
 for record in cases+errors:
  arguments=record['input']['arguments']
  if 'duration' in arguments:scalar(arguments['duration'],'PythonString')
  else:scalar(1,'PythonString')
  for raw in record['lines_hex']:
   try:evt=json.loads(bytes.fromhex(raw).decode().strip())
   except (ValueError,UnicodeError):continue
   if not isinstance(evt,dict):continue
   for key in ['title','code','message']:
    if key in evt:scalar(evt[key],'PythonString')
   for title in evt.get('titles',[]):scalar(title,'PythonString')
   if evt.get('duration_sec') is not None:scalar(evt['duration_sec'],'FixedOneDecimal')
   if evt.get('type')=='audio_chunk' and 'data' in evt:
    probe={'value':evt['data']}
    try:probe['bytes_hex']=base64.b64decode(evt['data']).hex()
    except Exception as error:probe['error_type']=type(error).__name__
    if probe not in base64_probes:base64_probes.append(probe)
  for timestamp in record['clock'][1:]:scalar(timestamp-record['clock'][0],'TruncatedInteger')
  if record['status']>=400:
   detail=record['error_json'].get('detail',{}) if isinstance(record['error_json'],dict) else None
   if isinstance(detail,dict) and 'message' in detail:detail=detail['message']
   if detail is not None:scalar(detail,'PythonString')
 for value in ['UNKNOWN','Unknown error',0,1,2,3,4,5]:scalar(value,'PythonString')
 return {'schema_version':1,'scalar_probes':scalar_probes,'base64_probes':base64_probes,'namespace':'zed.sim.provider.sonilo','scope':'actual source URLencoded request and NDJSON stream with injected canonical video/audio/clock owners; native allocation pending','source_sha256':{path:hashlib.sha256((SOURCE/path).read_bytes()).hexdigest() for path in ['sim_api_nodes/nodes_sonilo.py','sim_api_nodes/util/_helpers.py','sim_api_nodes/util/upload_helpers.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/conversions.py']},'urlencoded_wire_probes':wire_probes,'contracts':contracts,'cases':cases,'errors':errors,'aiohttp_version':aiohttp.__version__,'audio':{'bytes':len(WAV),'sha256':hashlib.sha256(WAV).hexdigest(),'producer':'stdlib wave PCM source fixture inherited from validated ElevenLabs; actual PyAV decode unavailable'},'video':{'bytes':len(MP4),'sha256':hashlib.sha256(MP4).hexdigest(),'producer':'pinned frontend plain_video.mp4; source save_to stub; no parse/codec proof'}}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();data=asyncio.run(generate());target=ROOT/'source-contract.json';encoded=json.dumps(data,indent=2,ensure_ascii=False)+'\n'
 if args.check:
  if target.read_text()!=encoded:raise ValueError('Sonilo source oracle differs')
 else:target.write_text(encoded)
 print('Sonilo actual source cases',len(data['cases']),'errors',len(data['errors']),'aiohttp',data['aiohttp_version'])
