import json
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw

cases=json.loads((Path(tempfile.gettempdir()) / 'sim-sdpose-native-face-review.json').read_text())
names=[('rectangle',1.0,False),('rectangle',1.5,False),('rectangle',1.5,True),('rectangle',10.0,True),('zero_score_and_negative_points',1.0,False),('zero_score_and_negative_points',1.5,True),('float32_normalization',1.5,True),('small_canvas',1.5,True)]
image=Image.new('RGB',(1280,720),'#f2f5f9');draw=ImageDraw.Draw(image)
for index,(name,scale,square) in enumerate(names):
 case=next(c for c in cases if c['id']==f'{name}-scale-{scale}-square-{square}')
 frame=case['frames'][0];box=case['actual'][0][0]
 left=20+(index%4)*320;top=20+(index//4)*350
 draw.text((left,top),name,fill='#18212c')
 draw.text((left,top+18),f'scale {scale} | square {square}',fill='#394759')
 w,h=frame['canvas_width'],frame['canvas_height'];factor=min(270/w,245/h)
 ox,oy=left+8,top+51;draw.rectangle((ox,oy,ox+w*factor,oy+h*factor),fill='white',outline='#b8c4d4',width=1)
 flat=frame['people'][0]['face_keypoints_2d']
 for point in range(0,len(flat),3):
  x,y,score=flat[point:point+3];px,py=ox+x*factor,oy+y*factor
  draw.ellipse((px-3,py-3,px+3,py+3),fill='#b453bd' if score==0 else '#1b63af')
 x,y,bw,bh=(box[k] for k in ['x','y','width','height'])
 draw.rectangle((ox+x*factor,oy+y*factor,ox+(x+bw)*factor,oy+(y+bh)*factor),outline='#0a966e',width=3)
 draw.text((left,top+311),f'box ({x}, {y}, {bw}, {bh})',fill='#18212c')
image.save(Path(__file__).with_name('native-face-boxes-review.png'))
