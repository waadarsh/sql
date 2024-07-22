import asyncio
import re
from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta

class SubprocessManager:
    def __init__(self):
        self.process = None
        self.status = "Not Started"
        self.progress = {"epoch": 0, "step": 0, "total_steps": 0, "loss": 0}

    async def run_script(self, command):
        self.status = "Running"
        self.progress = {"epoch": 0, "step": 0, "total_steps": 0, "loss": 0}
        try:
            self.process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break
                line = line.decode().strip()
                self.parse_progress(line)
                
            await self.process.wait()
            self.status = "Completed" if self.process.returncode == 0 else "Failed"
        except Exception as e:
            self.status = "Failed"
            raise e
        finally:
            self.process = None

    def parse_progress(self, line):
        # Adjust these patterns based on your actual output format
        epoch_pattern = r"Epoch (\d+)/(\d+)"
        step_pattern = r"Step (\d+)/(\d+)"
        loss_pattern = r"Loss: ([\d.]+)"

        if match := re.search(epoch_pattern, line):
            self.progress["epoch"] = int(match.group(1))
        if match := re.search(step_pattern, line):
            self.progress["step"] = int(match.group(1))
            self.progress["total_steps"] = int(match.group(2))
        if match := re.search(loss_pattern, line):
            self.progress["loss"] = float(match.group(1))

    def get_status(self):
        return self.status

    def get_progress(self):
        return self.progress

app = FastAPI()
subprocess_manager = SubprocessManager()
command = "accelerate launch train_dreambooth_lora_sdxl.py"

@app.get("/run")
async def run_subprocess():
    if subprocess_manager.get_status() == "Running":
        raise HTTPException(status_code=400, detail="A subprocess is already running.")
    
    # Start the subprocess asynchronously and respond immediately
    asyncio.create_task(subprocess_manager.run_script(command))
    return {"detail": "Subprocess started"}

@app.get("/status")
def subprocess_status():
    return {"status": subprocess_manager.get_status(), "progress": subprocess_manager.get_progress()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test-endpoint:app", host="127.0.0.1", port=8000, reload=False)
