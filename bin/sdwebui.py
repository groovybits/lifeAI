#!/usr/bin/env python

import webuiapi

# create API client with custom host, port
api = webuiapi.WebUIApi(
    host='127.0.0.1', 
    port=7860, 
    use_https=False)

models = api.util_get_model_names()

api.refresh_checkpoints()
print(f"Available models: {models}")
current_model = api.util_get_current_model()
print(f"Current model: {current_model}")

api.util_set_model('protogen')
api.util_set_model('dreamshaper')
api.util_set_model('sd_xl_turbo')

result = api.txt2img(prompt="trippy clouds",
                    negative_prompt="ugly, out of frame",
                    save_images=True,
                    width=512,
                    height=512,
#                    seed=1003,
                    #styles=["anime"],
#                    cfg_scale=7,
#                      sampler_index='DDIM',
#                      steps=30,
#                      enable_hr=True,
#                      hr_scale=2,
#                      hr_upscaler=webuiapi.HiResUpscaler.Latent,
#                      hr_second_pass_steps=20,
#                      hr_resize_x=1536,
#                      hr_resize_y=1024,
#                      denoising_strength=0.4,

          )

api.util_wait_for_ready()

if result.image is not None:
    result.image.save("out.png")
    print("Saved image to out.png")



