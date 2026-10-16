import asyncio
import pytest
from fastapi import HTTPException
from app.server import Engine, Request

def test_validation():
    with pytest.raises(Exception):Request(features=[0.1]*5)
@pytest.mark.asyncio
async def test_batch_and_canary():
    e=Engine('artifacts/model.pt',workers=2,batch_size=8,batch_wait_ms=3,canary_fraction=1)
    await e.start()
    try:
        outcomes=await asyncio.gather(*[e.infer([0.1]*32) for _ in range(20)])
        assert len(outcomes)==20
        assert all(v.version=='canary' and 0<=v.class_id<4 for v in outcomes)
        assert e.metrics()['completed']==20
    finally:await e.stop()
@pytest.mark.asyncio
async def test_reject_on_full_queue():
    e=Engine('artifacts/model.pt',workers=1,queue_limit=1)
    with pytest.raises(HTTPException) as exc:
        e.queue.put_nowait(object())
        await e.infer([0.0]*32)
    assert exc.value.status_code==503
