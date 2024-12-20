root_path=/workspace/stable-diffusion-webui
modify_path=/modify
cp -f $modify_path/modules/sd_vae_approx.py $root_path/modules/sd_vae_approx.py 
cp -f $modify_path/stable-diffusion-stability-ai/ldm/modules/encoders/modules.py $root_path/repositories/stable-diffusion-stability-ai/ldm/modules/encoders/modules.py
cp -f $modify_path/modules/devices.py $root_path/modules/devices.py 
cp -f $modify_path/server.py $root_path/server.py