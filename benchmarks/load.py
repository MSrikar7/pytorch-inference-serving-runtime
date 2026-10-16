import argparse, asyncio, json, statistics, time
import httpx
from app.server import app
async def run(count=600,concurrency=32):
    times=[];codes={};sem=asyncio.Semaphore(concurrency)
    async with app.router.lifespan_context(app):
        transport=httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,base_url='http://test') as client:
            async def one(i):
                async with sem:
                    start=time.perf_counter()
                    response=await client.post('/predict',json={'features':[float((i+j)%9)/9 for j in range(32)]})
                    times.append((time.perf_counter()-start)*1000)
                    codes[response.status_code]=codes.get(response.status_code,0)+1
            start=time.perf_counter()
            await asyncio.gather(*(one(i) for i in range(count)))
            duration=time.perf_counter()-start
            metrics=(await client.get('/metrics')).json()
    ordered=sorted(times)
    result={'load_test':'in-process ASGI; excludes network latency','requests':count,'concurrency':concurrency,'status_codes':codes,'throughput_rps':round(count/duration,2),'p50_ms':round(statistics.median(ordered),3),'p95_ms':round(ordered[int(.95*(count-1))],3),'p99_ms':round(ordered[int(.99*(count-1))],3),'engine':metrics}
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--requests',type=int,default=600);p.add_argument('--concurrency',type=int,default=32);p.add_argument('--out',default='results/load.json');args=p.parse_args()
    from pathlib import Path
    output=asyncio.run(run(args.requests,args.concurrency));p=Path(args.out);p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(output,indent=2))
