import io, os, threading
from functools import lru_cache
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

app=FastAPI(title='NutriVision AI',version='1.0.0')
app.mount('/static',StaticFiles(directory='static'),name='static')
MAX_BYTES=8*1024*1024
LABELS=['steamed white rice','yellow lentil dal','paneer curry','roti flatbread','vegetable curry','chicken curry','salad','biryani','boiled egg','fried fish','idli','samosa','plain yogurt','potato curry','unknown food']
# Approximate recipe-specific examples per 100g; not lab-verified or personalized.
NUTRITION={
 'steamed white rice':(130,2.7,28.2,.3),'yellow lentil dal':(115,6.5,15,3),
 'paneer curry':(240,11,8,18),'roti flatbread':(300,9,52,6),
 'vegetable curry':(110,3,12,6),'chicken curry':(190,19,6,10),
 'salad':(30,1.5,5,.4),'biryani':(190,6,29,6),
 'boiled egg':(155,13,1.1,11),'fried fish':(220,21,7,12),
 'idli':(130,4,27,.5),'samosa':(260,5,32,13),
 'plain yogurt':(65,3.5,4.7,3.3),'potato curry':(130,2,19,5)
}
_lock=threading.Lock()
@lru_cache(maxsize=1)
def models():
    from transformers import pipeline
    detector=pipeline('zero-shot-object-detection',model=os.getenv('DETECTOR_MODEL','google/owlv2-base-patch16-ensemble'),device=-1)
    classifier=pipeline('zero-shot-image-classification',model=os.getenv('CLIP_MODEL','openai/clip-vit-base-patch32'),device=-1)
    return detector,classifier

def classify(image, classifier):
    preds=classifier(image,candidate_labels=[f'a photo of {x}' for x in LABELS])
    best=preds[0]
    return best['label'].replace('a photo of ','',1),round(float(best['score']),3)

@app.get('/')
def index():return FileResponse('static/index.html')
@app.get('/health')
def health():return {'status':'ok','model_loaded':models.cache_info().currsize>0,'mode':'real-inference'}
@app.get('/api/foods')
def foods():return {'foods':[{'name':n,'per_100g':dict(zip(['kcal','protein','carbs','fat'],v))} for n,v in NUTRITION.items()], 'source':'illustrative recipe averages; verify against a food composition database'}

@app.post('/api/analyze')
async def analyze(file:UploadFile=File(...),description:str=Form('')):
    if file.content_type not in ('image/jpeg','image/png','image/webp'):
        raise HTTPException(415,'Upload a JPG, PNG, or WebP image')
    data=await file.read(MAX_BYTES+1)
    if len(data)>MAX_BYTES:raise HTTPException(413,'Maximum image size is 8 MB')
    try:
        image=ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
        if image.width*image.height>16_000_000:raise ValueError('Image too large')
    except (UnidentifiedImageError,ValueError,OSError):raise HTTPException(400,'Invalid or oversized image')
    image.thumbnail((1280,1280))
    try:
        with _lock:
            detector,classifier=models()
            prompts=[f'a photo of {x}' for x in LABELS if x!='unknown food']
            detections=detector(image,candidate_labels=prompts,threshold=.13)
            detections=sorted(detections,key=lambda d:float(d['score']),reverse=True)
            chosen=[]
            for d in detections:
                b=d['box'];box=[max(0,int(b[k])) for k in ('xmin','ymin','xmax','ymax')]
                box[2]=min(image.width,box[2]);box[3]=min(image.height,box[3])
                if box[2]-box[0]<18 or box[3]-box[1]<18:continue
                area=(box[2]-box[0])*(box[3]-box[1]);overlap=False
                for r in chosen:
                    a=r['box'];iw=max(0,min(a[2],box[2])-max(a[0],box[0]));ih=max(0,min(a[3],box[3])-max(a[1],box[1]));ia=iw*ih
                    if ia/min(area,(a[2]-a[0])*(a[3]-a[1]))>.65:overlap=True;break
                if overlap:continue
                crop=image.crop(tuple(box));name,score=classify(crop,classifier)
                chosen.append({'name':name,'confidence':score,'detection_score':round(float(d['score']),3),'box':box,'grams':None,'per_100g':dict(zip(['kcal','protein','carbs','fat'],NUTRITION[name])) if name in NUTRITION else None})
                if len(chosen)>=8:break
            if not chosen:
                name,score=classify(image,classifier)
                chosen=[{'name':name,'confidence':score,'detection_score':None,'box':None,'grams':None,'per_100g':dict(zip(['kcal','protein','carbs','fat'],NUTRITION[name])) if name in NUTRITION else None}]
                warning='No separate food regions confidently detected; this is a whole-image CLIP guess.'
            else:warning='Bounding boxes are detections, not segmentation masks. Food labels may be incorrect; confirm each item.'
        return {'items':chosen,'width':image.width,'height':image.height,'description':description[:300], 'warning':warning,'portion_method':'manual-confirmation-required','nutrition_source':'illustrative per-100g recipe averages'}
    except Exception as exc:
        raise HTTPException(503,'AI model unavailable. Check server logs, memory and model download.') from exc

class PortionItem(BaseModel):
    name:str
    grams:float=Field(ge=0,le=3000)
@app.post('/api/nutrition')
def nutrition(items:list[PortionItem]):
    result=[];totals=dict(kcal=0.,protein=0.,carbs=0.,fat=0.)
    for item in items:
        if item.name not in NUTRITION:raise HTTPException(422,f'Unknown food: {item.name}')
        n=dict(zip(totals,NUTRITION[item.name]));v={k:round(x*item.grams/100,2) for k,x in n.items()}
        for k in totals:totals[k]+=v[k]
        result.append({'name':item.name,'grams':item.grams,'nutrients':v})
    return {'items':result,'totals':{k:round(v,2) for k,v in totals.items()},'disclaimer':'Approximate recipe averages. Not suitable for clinical decisions.'}
