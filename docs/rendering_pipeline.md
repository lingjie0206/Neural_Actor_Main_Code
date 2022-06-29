# Pipeline for free-view human video synthesis with pose control

## Table of contents

  * [Data preparation](#data-preparation)
  * [Pre-trained models](#pretrained-models)
  * [Rendering with pose control](#rendering-with-pose-control)
  * [Quantitative Evaluation](#evaluation-for-self-reenactment-given-novel-poses)
------

## Data preparation

As a quick example, we can [download](https://dl.fbaipublicfiles.com/nsvf/neural_actor/example/sample_data.zip) and unzip the sample dataset. 
```bash
wget -O sample.zip https://dl.fbaipublicfiles.com/nsvf/neural_actor/example/sample_data.zip
unzip sample.zip -d workplace/
```
It shows the following structure:
```bash
<dataset name>
|-- canonical.obj         # a SMPL mesh of standard canonical pose
|-- transform_tpose.json  # a json file for transforming the standard canonical pose to a desired space
|-- uvmapping.obj         # a mesh saved uv coordinates to the texture map
|-- skinning_weight.txt   # skinning weights for each joint defined
|-- transform             # json files defined the target pose transformation (produced by EasyMocap) 
    |-- 000000.json       
    |-- 000001.json  
    ...
|-- intrinsics.txt        # camera intriniscs used for rendering
```
By default the data is saved under ``workplace/sample``.
To obtain the target pose files for your own dataset, we refer the readers to [EasyMocap](https://github.com/zju3dv/EasyMocap) to capture the performance information from videos.

## Pretrained models

Download the pretrained models of (1) [normal-texture translation](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/vid2vid/vid2vid_lan.pt) (2) [novel view synthesis](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/nerf/nerf_lan.pt) for the sample data. 
For instance,
```bash
texture_model=$PWD/workplace/vid2vid_lan.pt
rendering_model=$PWD/workplace/nerf_lan.pt

wget -O ${texture_model}   https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/vid2vid/vid2vid_lan.pt
wget -O ${rendering_model} https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/nerf/nerf_lan.pt
```


## Rendering with pose control

### STEP 1: generate normal maps

Similar to training, the first stage is to prepare the normal map information to predict the textures.
```bash
data_path=$PWD/workplace/sample
pushd texture_tool
python extract_normal.py \
    --mesh        ${data_path}/canonical.obj \
    --weights     ${data_path}/skinning_weight.txt \
    --texuv       ${data_path}/uvmapping.obj \
    --joint-data  ${data_path}/transform \
    --output-path ${data_path}/normal
popd
```
The output normal images wiil be saved in ``${data_path}/normal``, one image per target pose.

### STEP 2: predict texture maps

We use the pretrained image translation model to predict texture information:
```bash
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
```
The the predicted texture maps can be found as follows:


https://user-images.githubusercontent.com/5780274/172693409-cd97cead-550c-488a-9381-5d8fd83501e6.mp4



**Note that**: generation may fail when ``imageinaire`` failed download ``flownet2`` checkpoint. <br>
To fix this, please download [flownet](https://docs.google.com/uc?export=download&id=1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da) manually and save it under ``imaginaire/checkpoints/``.


### STEP 3: generate video given camera poses

We can then generate the target video based on the target poses and synthesized texture maps, and user specified camera poses.
Please follow the scripts:
```bash
# important to provide the initial files as an argument to override the saved paths
inference_args="{'texuv':'${data_path}/uvmapping.obj','new_tpose':'${data_path}/transform_tpose.json','mesh':'${data_path}/canonical.obj','weights':'${data_path}/skinning_weight.txt'}"
resolution="1024x1024"
fps=40
rotation_speed=0

# generate the video given a fixed camera
python render.py ${data_path} \
    --user-dir fairnr \
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
```
The code will automatically be launched using all avaiable GPUs, and save the output (images, video) in ``--render-output``.
We can also specify the output types by setting ``--render-output-types``, e.g. ``color`` means RGB image, ``normal`` means the normal image (generated based on depth), ``voxel`` means the textured mesh. The following shows the results:


https://user-images.githubusercontent.com/5780274/172693075-05fd64ab-9bb7-4dfa-9027-c6d5815b72a2.mp4


In this implementation, we set the camera parameters in the same format as [NSVF](https://github.com/facebookresearch/NSVF) with ``--render-at-vector``, ``--render-up-vector`` and ``--render-path-args``.
We can easily generate a free-view video by letting the camera move following the predefined trajectory. For instance, we can easily rotate the above example by setting ``rotation_speed=2``. Here is the result:

https://user-images.githubusercontent.com/5780274/172692937-18f5a21c-7ace-404e-8e8e-8c9e31533515.mp4


## Evaluation for self-reenactment given novel poses

Our code also support quantitative evaluation by rendering the character under an unseen poses, and compare the output with the ground-truth video. To do this, [download](https://dl.fbaipublicfiles.com/nsvf/neural_actor/example/sample_with_rgb.zip) and unzip the to ``${PWD}/workplace/sample_eval``. Similarly, it contains dataset in the following format:
```bash
<dataset name>
|-- canonical.obj         # a SMPL mesh of standard canonical pose
|-- transform_tpose.json  # a json file for transforming the standard canonical pose to a desired space
|-- uvmapping.obj         # a mesh saved uv coordinates to the texture map
|-- skinning_weight.txt   # skinning weights for each joint defined
|-- intrinisc             # camera intrinsics for each camera, fixed across all frames 
    |-- 0000.txt
    |-- 0001.txt
    ...
|-- pose                  # camera poses for each camera, fixed across all frames
    |-- 0000.txt
    |-- 0001.txt
    ...
|-- transform             # json files defined the target pose transformation (produced by EasyMocap) 
    |-- 000000.json       
    |-- 000001.json  
    ...
|-- rgb                   # ground-truth RGB image for each frame and each camera
    |-- 000000            # frame id
        |-- 0000.png
        |-- 0001.png
        ...
    |-- 000001            # camera id
        |-- 0000.png
        |-- 0001.png
        ...
    ...     
```

Then, follow [Step 1](#step-1-generate-normal-maps) and [Step 2](#step-2-predict-texture-maps) to generate texture maps.

### STEP 3b: generate video given the specific cameras and frames

In evaulation, we can manually set the target camera, the start & end frames, and the rendering frequency. The following is an example:
```bash
# important to provide the initial files as an argument to override the saved paths
inference_args="{'texuv':'${data_path}/uvmapping.obj','new_tpose':'${data_path}/transform_tpose.json','mesh':'${data_path}/canonical.obj','weights':'${data_path}/skinning_weight.txt'}"

resolution="1024x1024"
start_end="(200,400)"
subsample=1
camera_views="0,3,5,8"

python validate.py ${data_path} \
    --user-dir fairnr \
    --model-overrides ${inference_args} \
    --start-end ${start_end} \
    --valid-views ${camera_views} \
    --subsample-valid ${subsample} \
    --subsample-train ${subsample} \
    --valid-view-resolution ${resolution} \
    --load-video-dataset \
    --no-preload \
    --task single_object_rendering \
    --batch-size 1 \
    --valid-view-per-batch 1 \
    --path ${rendering_model} \
    --output-valid ${data_path}/eval
```
Here we specifically render a 200 frames (T=200...400), 4-view video based on the given cameras and novel poses (and the predicted texture).
The results are saved in ``${data_path}/eval``.

### STEP 4: compute the reconstruction scores

After running the ```validate.py``` code, it will automatically print out some numbers for the evaluation.
Optionally, you can also run the evaluation saperately:
```bash
python texture_tool/get_scores.py ${data_path}/eval/output ${data_path}/eval/target
```
You can also visualize the comparison (left: model output, right: ground-truth) by running the following code:
```bash
python texture_tool/concat_validation_video.py ${data_path}/eval
```


https://user-images.githubusercontent.com/5780274/172981248-c5182066-e81d-4498-b99c-fde805543a77.mp4

