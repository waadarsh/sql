import io
import os
import base64
from fastapi import FastAPI, WebSocket
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
from compel import Compel
import torch
import asyncio
from datetime import datetime
import json
from PIL import Image

app = FastAPI()

class ImageGenerationServer:
    def __init__(self):
        self.output_dir = "generated_images"
        os.makedirs(self.output_dir, exist_ok=True)

        self.generator = torch.Generator(device="cuda").manual_seed(8)

        self.base = StableDiffusionPipeline.from_pretrained(
            "test_sd_model_small",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.base.enable_model_cpu_offload()

        self.compel = Compel(
            tokenizer=self.base.tokenizer,
            text_encoder=self.base.text_encoder,
        )

        self.img2img = StableDiffusionImg2ImgPipeline(**self.base.components)
        self.base.scheduler = EulerAncestralDiscreteScheduler.from_config(self.base.scheduler.config)

        self.strength = 0.6
        self.num_inference_steps = 50
        self.guidance_scale = 5
        self.callback_steps = 1

    async def generate_image(self, conditioning, negative_conditioning, init_image, websocket: WebSocket):
        loop = asyncio.get_running_loop()

        def progress_callback(step, t, latents):
            progress = int((step / self.num_inference_steps) * 100)
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({"progress": progress, "status": "Generating"}),
                loop
            )

        with torch.no_grad():
            if init_image:
                init_image = Image.open(io.BytesIO(base64.b64decode(init_image.split(",")[1])))
                image = await loop.run_in_executor(None, 
                    lambda: self.img2img(
                        prompt_embeds=conditioning,
                        image=init_image,
                        negative_prompt_embeds=negative_conditioning,
                        strength=self.strength,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        callback=progress_callback,
                        callback_steps=self.callback_steps,
                    )
                )
            else:
                image = await loop.run_in_executor(None, 
                    lambda: self.base(
                        prompt_embeds=conditioning,
                        negative_prompt_embeds=negative_conditioning,
                        generator=self.generator,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        callback=progress_callback,
                        callback_steps=self.callback_steps,
                    )
                )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_image_{timestamp}.png"
        save_path = os.path.join(self.output_dir, filename)
        image.images[0].save(save_path)

        img_byte_arr = io.BytesIO()
        image.images[0].save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        await websocket.send_json({"progress": 100, "status": "Generation Completed"})
        await websocket.send_bytes(img_byte_arr)
        await websocket.send_json({"save_path": save_path})

server = ImageGenerationServer()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message["type"] == "prompt":
                prompt = message["prompt"]
                init_image = message.get("image")
                negative_prompt = message.get("negative_prompt")

                conditioning = server.compel([prompt])
                negative_conditioning = server.compel([negative_prompt]) if negative_prompt else None

                await server.generate_image(conditioning, negative_conditioning, init_image, websocket)
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000, reload=False)
