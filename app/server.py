"""Multi-worker inference service: adaptive admission, version routing and observability.
Workers are independent asyncio tasks sharing one process, not separate hosts.
"""
import asyncio, os, random, time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List
import torch
from torch import nn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

FEATURES=32
class Request(BaseModel):
    features: List[float]=Field(min_length=FEATURES,max_length=FEATURES)
class Prediction(BaseModel):
    class_id:int
    version:str
    batch_size:int
    latency_ms:float
@dataclass
class Item:
    x:list
    future:asyncio.Future
    created:float
    version:str

def make_model():
    return nn.Sequential(nn.Linear(32,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,4)).eval()
class Engine:
    def __init__(self,model_path,workers=2,batch_size=32,batch_wait_ms=5,queue_limit=512,canary_fraction=0.0):
        torch.set_num_threads(1)
        self.models={}
        base=make_model()
        if Path(model_path).exists():
            try: state=torch.load(model_path,map_location='cpu',weights_only=True)
            except TypeError: state=torch.load(model_path,map_location='cpu')
            base.load_state_dict(state['state_dict'])
        else: raise FileNotFoundError('Train model first: python training/train.py')
        self.models['stable']=base
        canary=make_model();canary.load_state_dict(base.state_dict());self.models['canary']=canary
        self.queue=asyncio.Queue(maxsize=queue_limit)
        self.worker_count=workers;self.batch_size=batch_size;self.wait=batch_wait_ms/1000
        self.canary_fraction=canary_fraction;self.tasks=[]
        self.accepted=0;self.completed=0;self.rejected=0;self.cancelled=0;self.failed=0;self.batches=0
        self.latencies=deque(maxlen=20000);self.version_count={'stable':0,'canary':0}
    async def start(self):
        self.tasks=[asyncio.create_task(self.work_loop()) for _ in range(self.worker_count)]
    async def stop(self):
        for task in self.tasks:task.cancel()
        await asyncio.gather(*self.tasks,return_exceptions=True)
        while not self.queue.empty():
            item=self.queue.get_nowait();self.queue.task_done()
            if not item.future.done():item.future.cancel()
    async def infer(self,x):
        f=asyncio.get_running_loop().create_future()
        version='canary' if random.random()<self.canary_fraction else 'stable'
        item=Item(x,f,time.perf_counter(),version)
        try:self.queue.put_nowait(item)
        except asyncio.QueueFull:
            self.rejected+=1
            raise HTTPException(503,'Inference queue saturated')
        self.accepted+=1
        try:return await asyncio.wait_for(f,timeout=10.0)
        except asyncio.TimeoutError:
            f.cancel();self.cancelled+=1
            raise HTTPException(504,'Inference timed out')
        except asyncio.CancelledError:
            f.cancel();self.cancelled+=1;raise
    async def work_loop(self):
        while True:
            first=await self.queue.get();batch=[first]
            deadline=time.perf_counter()+self.wait
            while len(batch)<self.batch_size:
                remaining=deadline-time.perf_counter()
                if remaining<=0:break
                try:batch.append(await asyncio.wait_for(self.queue.get(),remaining))
                except asyncio.TimeoutError:break
            try:
                for version in ('stable','canary'):
                    selected=[i for i in batch if i.version==version and not i.future.cancelled()]
                    if not selected:continue
                    data=torch.tensor([i.x for i in selected],dtype=torch.float32)
                    with torch.inference_mode():outputs=self.models[version](data).argmax(dim=1).tolist()
                    for item,pred in zip(selected,outputs):
                        latency=(time.perf_counter()-item.created)*1000
                        if not item.future.done():
                            item.future.set_result(Prediction(class_id=pred,version=version,batch_size=len(batch),latency_ms=round(latency,4)))
                            self.completed+=1;self.version_count[version]+=1;self.latencies.append(latency)
                self.batches+=1
            except Exception as exc:
                self.failed+=len(batch)
                for item in batch:
                    if not item.future.done():item.future.set_exception(exc)
            finally:
                for _ in batch:self.queue.task_done()
    def metrics(self):
        values=sorted(self.latencies)
        percentile=lambda p:round(values[min(len(values)-1,int((len(values)-1)*p))],3) if values else 0
        return {'accepted':self.accepted,'completed':self.completed,'rejected':self.rejected,'cancelled':self.cancelled,'failed':self.failed,'batches':self.batches,'queue_depth':self.queue.qsize(),'worker_tasks':self.worker_count,'canary_fraction':self.canary_fraction,'versions':self.version_count,'p50_ms':percentile(.5),'p95_ms':percentile(.95),'p99_ms':percentile(.99)}
@asynccontextmanager
async def lifespan(app):
    app.state.engine=Engine(os.getenv('MODEL_PATH','artifacts/model.pt'),workers=int(os.getenv('WORKERS','2')),batch_size=int(os.getenv('MAX_BATCH','32')),batch_wait_ms=float(os.getenv('BATCH_WAIT_MS','5')),queue_limit=int(os.getenv('QUEUE_LIMIT','512')),canary_fraction=float(os.getenv('CANARY_FRACTION','0.1')))
    await app.state.engine.start()
    try:yield
    finally:await app.state.engine.stop()
app=FastAPI(title='Systems ML Inference Runtime',lifespan=lifespan)
@app.post('/predict',response_model=Prediction)
async def predict(payload:Request):return await app.state.engine.infer(payload.features)
@app.get('/health')
async def health():return {'status':'ready'}
@app.get('/metrics')
async def metrics():return app.state.engine.metrics()
@app.get('/metrics/prometheus')
async def prometheus():
    m=app.state.engine.metrics();return Response(''.join(f'systems_ml_{k} {v}\n' for k,v in m.items() if isinstance(v,(int,float))),media_type='text/plain')
