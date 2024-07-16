# main.py
import asyncio
from fastapi import FastAPI, HTTPException

class SubprocessManager:
    def __init__(self):
        self.process = None
        self.status = "Not Started"

    async def run_script(self, command):
        self.status = "Running"
        self.process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await self.process.communicate()
        self.status = "Completed" if self.process.returncode == 0 else "Failed"
        return stdout.decode(), stderr.decode()

    def get_status(self):
        return self.status

app = FastAPI()
subprocess_manager = SubprocessManager()

command = """
autotrain dreambooth \
--model='stabilityai/stable-diffusion-xl-base-1.0' \
--project-name='lora' \
--image-path='/content/images' \
--prompt='a photo of nissan_concept' \
--resolution=1024 \
--batch-size=1 \
--num-steps=500 \
--gradient-accumulation=4 \
--lr=1e-4 \
--mixed-precision fp16 \
--xformers
"""

@app.get("/run-subprocess")
async def run_subprocess():
    if subprocess_manager.get_status() == "Running":
        raise HTTPException(status_code=400, detail="A subprocess is already running.")
    stdout, stderr = await subprocess_manager.run_script(command)
    return {"stdout": stdout, "stderr": stderr}

@app.get("/subprocess-status")
def subprocess_status():
    return {"status": subprocess_manager.get_status()}
