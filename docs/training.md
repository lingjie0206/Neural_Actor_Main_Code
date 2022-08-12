# Training a Neural Actor model

## Table of contents

  * [Data preparation](#data-preparation)
  * [Learn a texture prediction model](#learn-texture-prediction)
  * [Learning a neural rendering model](#learn-neural-rendering)
------

## Data preparation

Please follow the [steps](../README.md#dataset) to request the full datasets. Each dataset has the following structure (testing/training sets):
```bash
<dataset_path>
|-- intrinisc         
    |-- 0000.txt
    |-- 0001.txt
    ...
|-- pose             
    |-- 0000.txt
    |-- 0001.txt
    ...
|-- testing
    |-- rgb_video
        |-- 000.avi
        |-- 001.avi
        ...
    |-- normal_modifysmpluv0.1_smooth3e-2.avi
    |-- tex_modifysmpluv0.1_smooth3e-2.avi
    |-- transform.zip
|-- training
    |-- rgb_video
        |-- 000.avi
        |-- 001.avi
        ...
    |-- normal_modifysmpluv0.1_smooth3e-2.avi
    |-- tex_modifysmpluv0.1_smooth3e-2.avi
    |-- transform.zip
```

We provide the script to extract images from videos, and get the downloaded dataset into following format. For example,
```bash
python scripts/run_ffmpeg_uncompress.py -i <dataset_path>/training/ -o <dataset_path>/training_images -f 0,1000  # start_frame=0, end_frame=1000

# copy the camera parameters
cp -r <dataset_path>/pose <dataset_path>/training_images/
cp -r <dataset_path>/intrinisc <dataset_path>/training_images/
```
We can specify the desired sequence from the video file by setting ``-f start,end``.

Also download the additional files from [Google drive](https://drive.google.com/drive/folders/1cXk623v7p1eo9566tuxE8hXWIF9ov8hR?usp=sharing) as [stated](../README.md#dataset), and put them together with the extracted files as follows:
```bash
<dataset_path>/training_images
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
|-- normal                # generated normal maps from the poses
    |-- 000000.png
    |-- 000001.png
    ...
|-- tex                   # ground-truth texture images obtained from real rgb images
    |-- 000000.png
    |-- 000001.png
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

## Learn Texture Prediction
Our texture prediction model is based on [vid2vid](https://tcwang0509.github.io/vid2vid/).
For more details, please follow the original [README](https://github.com/MultiPath/imaginaire-stable/blob/3c6b784a91456ed11d493eb57935f0679ca4d6ee/projects/vid2vid/README.md) for dataset preparation and training arguments.
Due to the inflexibility of the imaginaire implementation, we have to link the dataset folders to ``images`` and ``seg_maps`` before training. 

For example, we train a vid2vid texture map predictor from the normal map sequence as follows:
* Prepare dataset
```bash
mkdir -p <dataset_path>/training_images/vid2vid
cd <dataset_path>/training_images/vid2vid
ln -s ../tex images         # softlink texture images as target
ln -s ../normal seg_maps    # softlink normal images as input
```
* Train on single GPU
```bash
pushd imaginaire
python train.py --config ../vid2vid.yaml --single_gpu --traindir <dataset_path>/training_images/vid2vid --logdir <output_path>

# or training with multi-GPUs
# python -m torch.distributed.launch --nproc_per_node=N train.py --config ../vid2vid.yaml --traindir <dataset_path>/training_images/vid2vid --logdir <output_path>
popd 
```
The learned model and log will be saved in ``<output_path>``.

## Learn Neural Rendering
Our neural renderer is trained given the ground-truth texture maps and tracked pose information. 
The following is an example to train a default neural actor model over the processed dataset (lan):
```bash
# data/checkpoint settings
DATA="lan"
RES="1024x1024"
DATASET=workplace/${DATA}/training_images
MODEL="joint_nerf_coarse"
SUFFIX="debug"
ARCH=${DATA}_${MODEL}_${SUFFIX}
SAVE=workplace/checkpoints
mkdir -p $SAVE/$ARCH

# hyperparameters
COMMON_FLAGS="--user-dir fairnr --task single_object_rendering --no-sampling-at-reader --no-preload --load-video-dataset --num-workers 0 --broadcast-buffers"
DATA_FLAGS="--train-views 0..11 --view-resolution ${RES} --valid-views 0,3,7,10 --valid-view-resolution ${RES} --subsample-valid 200 --subsample-train 1"
INPUT_FLAGS="--mesh ${DATASET}/canonical.obj --weights ${DATASET}/skinning_weight.txt --texuv ${DATASET}/uvmapping.obj --new-tpose ${DATASET}/transform_tpose.json"
ENCODER_FLAGS="--min-dis-eps 0.06 --use-local-coordinate --additional-deform pos --texture-layers 3 --texture-to-deformation --use-texture-encoder"
NERF_FLAGS="--fixed-num-samples 64 --inputs-to-texture feat:0:256,attnout:0:256,texture:0:512,ray:4:3:b --transparent-background 1.0,1.0,1.0 --background-stop-gradient --discrete-regularization"
LOSS_FLAGS="--color-weight 1.0 --alpha-weight 0.01 --criterion srn_loss"
OPTIMIZER_FLAGS="--optimizer adam --adam-betas (0.9,0.999) --lr-scheduler exp --decay-steps 250000 --lr 0.0002 --warmup-updates 1 --clip-norm 0.0" 
CHECKPOINT_FLAGS="--save-interval-updates 500 --max-update 300000 --virtual-epoch-steps 5000 --save-interval 1 --keep-interval-updates 5 --keep-last-epochs 5 --no-epoch-checkpoints"
TRAIN_FLAGS="--batch-size 1 --view-per-batch 1 --pixel-per-view 1024 --chunk-size 256"
LOG_FLAGS="--log-interval 10 --log-format json --tensorboard-logdir ${SAVE}/tensorboard/${ARCH} --save-dir ${SAVE}/${ARCH}"

# launch training
python train.py ${DATASET} --arch $MODEL --seed 2 \
    $COMMON_FLAGS $DATA_FLAGS $INPUT_FLAGS $ENCODER_FLAGS $NERF_FLAGS $LOSS_FLAGS $OPTIMIZER_FLAGS $TRAIN_FLAGS $CHECKPOINT_FLAGS $LOG_FLAGS
```