root_path=/workspace/MaskGCT
modify_path=/modify
cp -f $modify_path/app.py $root_path/app.py 
cp -f $modify_path/vocos.py $root_path/models/codec/amphion_codec/vocos.py
cp -f $modify_path/main.py $root_path/main.py