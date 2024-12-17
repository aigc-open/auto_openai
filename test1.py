import webuiapi
from pydantic import BaseModel
from typing import Optional



api = webuiapi.WebUIApi(host='127.0.0.1', port=30000)
api.util_set_model(name="majicmixRealistic_v7.safetensors/majicmixRealistic_v7.safetensors")
print(api.get_sd_models())
# print(api.get_schedulers())
# print(api.get_samplers())
# print(api.controlnet_module_list())
# print(api.controlnet_model_list())
# api.txt2img
# api.img2img
# result1 = api.txt2img(
#                     prompt="cute squirrel",
#                     negative_prompt="ugly, out of frame",
#                     seed=1003,
#                     hr_checkpoint_name="majicmixRealistic_v7.safetensors/majicmixRealistic_v7.safetensors",
#                     styles=["anime"],
#                     cfg_scale=7,
# #                      sampler_index='DDIM',
# #                      steps=30,
# #                      enable_hr=True,
# #                      hr_scale=2,
# #                      hr_upscaler=webuiapi.HiResUpscaler.Latent,
# #                      hr_second_pass_steps=20,
# #                      hr_resize_x=1536,
# #                      hr_resize_y=1024,
# #                      denoising_strength=0.4,

#                     )
# print(result1)
# # images contains the returned images (PIL images)
# print(result1.images)

# # image is shorthand for images[0]
# print(result1.image)

# # info contains text info about the api call
# print(result1.info)

# # info contains paramteres of the api call
# print(result1.parameters)
