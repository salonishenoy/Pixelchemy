from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from torch import autocast
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64 


app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

device = "cpu"
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
pipe.to(device)

@app.get("/")
def generate(prompt: str): 
    with autocast(device): 
        image = pipe(prompt, guidance_scale=5).images[0]

    image.save("testimage.png")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    return Response(content=imgstr, media_type="image/png")
