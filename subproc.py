import asyncio
from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta

class SubprocessManager:
    def __init__(self):
        self.process = None
        self.status = "Not Started"
        self.start_time = None

    async def run_script(self, command):
        self.status = "Running"
        self.start_time = datetime.now()
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
            self.start_time = None

    def get_status(self):
        elapsed_time = None
        if self.start_time:
            elapsed_time = datetime.now() - self.start_time
        return {"status": self.status, "elapsed_time": str(elapsed_time) if elapsed_time else None}

app = FastAPI()
subprocess_manager = SubprocessManager()

command = "python subscript.py"

@app.get("/run")
async def run_subprocess():
    if subprocess_manager.get_status()["status"] == "Running":
        raise HTTPException(status_code=400, detail="A subprocess is already running.")
    
    # Start the subprocess asynchronously and respond immediately
    asyncio.create_task(subprocess_manager.run_script(command))
    return {"detail": "Subprocess started"}

@app.get("/check")
def subprocess_status():
    return subprocess_manager.get_status()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test-endpoint:app",host="127.0.0.1",port=8000,reload=False)
