from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import asyncio
from diffusers import StableDiffusionPipeline
import torch

app = FastAPI()
progress = 0

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

@app.get("/")
async def get():
    return HTMLResponse("Image Generation Progress API")

@app.get("/progress")
async def get_progress():
    return {"progress": progress}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global progress
    await websocket.accept()
    
    # Example prompt
    prompt = "A beautiful landscape painting"
    
    # Function to generate images with progress callback
    async def progress_callback(step, t, latents):
        global progress
        progress = int((step / pipe.scheduler.config.num_train_timesteps) * 100)
        await websocket.send_json({"progress": progress})
    
    # Generate image
    async with torch.no_grad():
        image = pipe(prompt, callback=progress_callback).images[0]
    
    await websocket.close()
