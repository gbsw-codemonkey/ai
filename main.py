!pip install fastapi uvicorn pyngrok diffusers transformers accelerate
!pip install git+https://github.com/huggingface/diffusers.git

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import uvicorn
from pyngrok import ngrok
import asyncio

app = FastAPI()

# 파이프라인 설정
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

class VideoRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 9.0

@app.post("/generate-video")
async def generate_video(request: VideoRequest):
    try:
        # 비디오 생성
        video_frames = pipe(
            request.prompt, 
            num_inference_steps=request.num_inference_steps, 
            guidance_scale=request.guidance_scale
        ).frames

        # 필요한 변환 적용
        video_frames = video_frames.squeeze(0)
        video_frames = (video_frames * 255).astype("uint8")

        # 색상 향상
        video_frames = np.clip(video_frames * 1.2, 0, 255).astype("uint8")

        # 비디오 저장
        video_path = export_to_video(video_frames)

        return {"video_path": video_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ngrok 설정 (이미 인증을 완료했으므로 이 줄은 주석)
# !ngrok config add-authtoken 2j27vD2VtJOyWNLlG1Hhe6aUTVl_782M5FWdcUq833RhR4ZhE

# ngrok 터널 생성
ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)

# FastAPI 애플리케이션 실행
if __name__ == "__main__":
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    
    # Use asyncio.create_task to run the server within the existing event loop
    asyncio.create_task(server.serve())
