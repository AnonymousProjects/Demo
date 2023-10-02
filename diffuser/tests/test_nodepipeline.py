from os import pipe2
import sys
sys.path.append("./src")
from diffusers import StableDiffusionNODEPipeline
from diffusers import NeuralODEScheduler

pipe = StableDiffusionNODEPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
scheduler = NeuralODEScheduler.from_config(pipe.scheduler.config)
#scheduler.prediction_type = "sample"


# from diffusers import StableDiffusionPipeline
# from diffusers import DPMSolverSinglestepScheduler

# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
# scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)

pipe.scheduler = scheduler
pipe.safety_checker = None

pipe.to('cuda')

prompt = "A photograph of a cute cat"

image = pipe(prompt).images[0]

image.save('/opt/tiger/_debug_/diffusers-main/prompt-tuning-test/sample7.jpg')