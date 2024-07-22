import asyncio
from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta

class SubprocessManager:
    def __init__(self):
        self.process = None
        self.status = "Not Started"

    async def run_script(self, command):
        self.status = "Running"
        try:
            self.process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await self.process.communicate()
            self.status = "Completed" if self.process.returncode == 0 else "Failed"
            return stdout.decode(), stderr.decode()
        except Exception as e:
            self.status = "Failed"
            raise e
        finally:
            self.process = None

    def get_status(self):
        return self.status

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
    return {"detail" : subprocess_manager.get_status()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test-endpoint:app",host="127.0.0.1",port=8000,reload=False)
