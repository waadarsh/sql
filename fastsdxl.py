import io
import os
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from diffusers import StableDiffusionPipeline
import torch
import asyncio
from datetime import datetime

app = FastAPI()
progress = 0
pipe = StableDiffusionPipeline.from_pretrained(
    "OFA-Sys/small-stable-diffusion-v0",
    torch_dtype=torch.float16,
    cache_dir="models",
).to("cuda")
pipe.enable_model_cpu_offload()

# Ensure the output directory exists
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Stable Diffusion Progress</title>
    </head>
    <body>
        <h1>Stable Diffusion Progress</h1>
        <p id="progress">Waiting for progress...</p>
        <img id="generated-image" style="display:none;" />
        <p id="save-path"></p>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.binaryType = "arraybuffer";
            ws.onmessage = function(event) {
                if (typeof event.data === "string") {
                    var data = JSON.parse(event.data);
                    if (data.progress !== undefined) {
                        document.getElementById('progress').innerText = 'Progress: ' + data.progress + '%';
                    } else if (data.save_path) {
                        document.getElementById('save-path').innerText = 'Image saved at: ' + data.save_path;
                    }
                } else {
                    document.getElementById('progress').innerText = 'Image generated!';
                    var blob = new Blob([event.data], {type: "image/png"});
                    var url = URL.createObjectURL(blob);
                    document.getElementById('generated-image').src = url;
                    document.getElementById('generated-image').style.display = 'block';
                }
            };
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

async def generate_image(prompt, websocket: WebSocket):
    loop = asyncio.get_running_loop()
    
    def progress_callback(step, t, latents):
        global progress
        progress = int((step % pipe.scheduler.config.num_train_timesteps) * 10)
        print(f"\n progress: {progress}")
        loop.create_task(websocket.send_json({"progress": progress}))

    with torch.no_grad():
        image = await asyncio.to_thread(pipe, prompt, callback=progress_callback, callback_steps=1)
    
    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_image_{timestamp}.png"
    save_path = os.path.join(output_dir, filename)
    
    # Save the image
    image.images[0].save(save_path)
    
    # Convert image to bytes for sending over WebSocket
    img_byte_arr = io.BytesIO()
    image.images[0].save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Send the image as binary
    await websocket.send_bytes(img_byte_arr)
    
    # Send the save path
    await websocket.send_json({"save_path": save_path})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    prompt = "A beautiful landscape painting"
    try:
        await generate_image(prompt, websocket)
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sdxl-test-1:app", host="127.0.0.1", port=8000, reload=False)
