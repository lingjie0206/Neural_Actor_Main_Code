# download
wget -O sample.zip https://dl.fbaipublicfiles.com/nsvf/neural_actor/example/sample_data.zip
unzip -q sample.zip -d workplace/

texture_model=$PWD/workplace/vid2vid_lan.pt
rendering_model=$PWD/workplace/nerf_lan.pt

# wget -O ${texture_model}   https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/vid2vid/vid2vid_lan.pt
# wget -O ${rendering_model} https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/nerf/nerf_lan.pt


# SREP 1
data_path=$PWD/workplace/sample
pushd texture_tool
python extract_normal.py \
    --mesh        ${data_path}/canonical.obj \
    --weights     ${data_path}/skinning_weight.txt \
    --texuv       ${data_path}/uvmapping.obj \
    --joint-data  ${data_path}/transform \
    --output-path ${data_path}/normal
popd

# STEP 2
# to meet imaginaire's requirement, we need to prepare ``seg_maps`` and ``images`` folders:
ln -s ${data_path}/normal ${data_path}/seg_maps 
ln -s ${data_path}/normal ${data_path}/images 

# generating texture maps using N GPUs (e.g., N=8 at our cluster)
pushd imaginaire
python -m torch.distributed.launch --nproc_per_node=8 inference.py \
    --config vid2vid_inference.yaml \
    --checkpoint ${texture_model} \
    --testdir ${data_path} \
    --output_dir ${data_path}/tex
popd


# remove unused files
rm -rf ${data_path}/tex/normal
rm -rf ${data_path}/seg_maps
rm -rf ${data_path}/images 

# STEP3
inference_args="{'texuv':'${data_path}/uvmapping.obj','new_tpose':'${data_path}/transform_tpose.json','mesh':'${data_path}/canonical.obj','weights':'${data_path}/skinning_weight.txt'}"
resolution="1024x1024"
fps=40
rotation_speed=0

# generate the video given a fixed camera
python render.py ${data_path} \
    --task single_object_rendering \
    --path ${rendering_model} \
    --batch-size 1 \
    --load-video-dataset \
    --model-overrides ${inference_args} \
    --render-beam 1 \
    --render-save-fps $fps \
    --render-angular-speed ${rotation_speed} \
    --render-num-frames 1 \
    --render-at-vector "(-0.43043275,  0.83874, -0.03646531)" \
    --render-up-vector "(0,-1,0)" \
    --render-path-args "{'radius': 2.82, 'axis': 'y', 'h': 0.6, 't0': 0}" \
    --render-resolution ${resolution} \
    --render-output ${data_path}/output \
    --render-output-types "color" "normal" \
    --render-combine-output

echo "done: ${data_path}/output"