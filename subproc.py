import asyncio
import re
from fastapi import FastAPI, HTTPException

class TrainingProgress:
    def __init__(self):
        self.current_step = 0
        self.total_steps = 50  # This matches the max_train_steps in the command
        self.status = "Not Started"

progress = TrainingProgress()
app = FastAPI()

async def run_training_script():
    global progress
    progress.status = "Running"
    command = """
    accelerate launch --config_file accelerate_config.yaml train_dreambooth_lora_sdxl.py \
      --pretrained_model_name_or_path='base'  \
      --instance_data_dir='new_concept' \
      --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
      --output_dir="Test_lora_1" \
      --mixed_precision="fp16" \
      --instance_prompt="a photo of sks nissan_concept" \
      --resolution=512 \
      --train_batch_size=4 \
      --gradient_accumulation_steps=2 \
      --gradient_checkpointing \
      --learning_rate=1e-4 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --max_train_steps=50 \
      --seed="0" \
    """
    
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    while True:
        line = await process.stdout.readline()
        if not line:
            break
        line = line.decode().strip()
        print(f"Debug: {line}")
        if match := re.search(r"Steps:\s*(\d+)%\|.+?\|\s*(\d+)/(\d+)", line):
            percentage = int(match.group(1))
            current_step = int(match.group(2))
            total_steps = int(match.group(3))
            progress.current_step = current_step
            progress.total_steps = total_steps
            print(f"Updated progress: {progress.current_step}/{progress.total_steps}")

    await process.wait()
    progress.status = "Completed" if process.returncode == 0 else "Failed"

@app.post("/start-training")
async def start_training():
    if progress.status == "Running":
        raise HTTPException(status_code=400, detail="Training is already in progress")
    asyncio.create_task(run_training_script())
    return {"message": "Training started"}

@app.get("/training-progress")
async def get_training_progress():
    return {
        "status": progress.status,
        "current_step": progress.current_step,
        "total_steps": progress.total_steps
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
