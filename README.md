# Neural Actor: Neural Free-view Synthesis of Human Actors with Pose Control

### [Project Page](https://vcai.mpi-inf.mpg.de/projects/NeuralActor/) | [Video](https://vcai.mpi-inf.mpg.de/projects/NeuralActor/) | [Paper](https://arxiv.org/abs/2106.02019) | [Data](#dataset)

<img src='docs/figs/first_img.png'/>

Abstract: *We propose Neural Actor (NA), a new method for high-quality synthesis of humans from arbitrary viewpoints and under arbitrary controllable poses. Our method is built upon recent neural scene representation and rendering works which learn representations of geometry and appearance from only 2D images. While existing works demonstrated compelling rendering of static scenes and playback of dynamic scenes, photo-realistic reconstruction and rendering of humans with neural implicit methods, in particular under user-controlled novel poses, is still difficult. To address this problem, we utilize a coarse body model as the proxy to unwarp the surrounding 3D space into a canonical pose. A neural radiance field learns pose-dependent geometric deformations and pose- and view-dependent appearance effects in the canonical space from multi-view video input. To synthesize novel views of high fidelity dynamic geometry and appearance, we leverage 2D texture maps defined on the body model as latent variables for predicting residual deformations and the dynamic appearance. Experiments demonstrate that our method achieves better quality than the state-of-the-arts on playback as well as novel pose synthesis, and can even generalize well to new poses that starkly differ from the training poses. Furthermore, our method also supports body shape control of the synthesized results.*


This is the official repository of "Neural Actor: Neural Free-view Synthesis of Human Actors with Pose Control" (SIGGRAPH Asia 2021)

## Table of contents
-----
  * [Installation](#requirements-and-installation)
  * [Dataset](#dataset)
  * [Usage](#train-a-new-model)
    + [Free-view Rendering Pipeline](#free-viewpoint-rendering-pipeline)
    + [Training](#train-a-new-model)
  * [License](#license)
  * [Citation](#citation)
------

## System requirements

This code is implemented based on [Neural Sparse Voxel Fields (NSVF)](https://github.com/facebookresearch/NSVF).
The code has been tested on the following system:

* Python 3.8
* PyTorch 1.6.0, torchvision 0.7.0
* [Nvidia apex library](https://github.com/NVIDIA/apex) (optional)
* Nvidia GPU (Tesla V100 32GB) CUDA 10.2

(**Please note that our codebase theoritically also work on higher PyTorch version >= 1.7.
However, due to some unexpected behavior changes in Conv2d at Pytorch 1.7, our pre-trained models would produce bad results.**)

Only learning and rendering on GPUs are supported.

## Installation

To install, first clone this repo and install all dependencies. We suggest installing in a virtual environment.

```bash
conda create -n neuralactor python=3.8
conda activate neuralactor

pip install -r requirements.txt
```
This code also depends on ``opendr``, which can be installed from source
```
mkdir -p tools
git clone https://github.com/MultiPath/opendr.git tools/opendr

pushd tools/opendr
python setup.py install
popd
```

Then,  run

```bash
pip install --editable .
```

Or if you want to install the code locally, run:

```bash
python setup.py build_ext --inplace
```

To train and generate texture maps, it also needs to compile ```imaginaire``` following the [instruction](https://github.com/NVlabs/imaginaire/blob/master/INSTALL.md).
You can also run:
```bash
# please make sure CUDA_HOME is set correctly before compiling.
export CUDA_HOME=/usr/local/cuda-10.2/   # or the path you installed CUDA
git submodule update --init

pushd imaginaire
bash scripts/install.sh
popd
```

Generation with ``imageinaire`` also requires a pretrained checkpoint of ``flownet2``. We suggest manually download before running.
```
mkdir -p imaginaire/checkpoints
gdown 1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da -O imaginaire/checkpoints/flownet2.pth.tar
```

## Dataset
Please find the character mapping in our paper:

![image](docs/figs/dataset-1.png)

For full datasets, please register through this [link](https://gvv-assets.mpi-inf.mpg.de/NeuralActor/?page_id=11#038;redirect_to=https%3A%2F%2Fgvv-assets.mpi-inf.mpg.de%2FNeuralActor%2F) by accepting agreements.

Each dataset also needs additional files which defines the canonical geometry, the UV parameterization and skinning weights. Please download from [Google drive folder](https://drive.google.com/drive/folders/1cXk623v7p1eo9566tuxE8hXWIF9ov8hR?usp=sharing)

We also provide the SMPL tracking results (pose and shape parameters) of each sequence. Please download from [Google drive folder](https://drive.google.com/drive/folders/1C5W4l3r2Rkewz84roqzEepQimBtPeHBY?usp=sharing)



## Prepare your own dataset

Please refer to a [saperate repo](https://github.com/lingjie0206/Neural_Actor_Preprocessing) for detailed steps for pre-processing your own datasets
<!-- To prepare a new dataset of a single scene for training and testing, please follow the data structure: -->



## Pre-trained Models

Character | Texture Predictor | Neural Renderer | Character | Texture Predictor | Neural Renderer
:---:|---|---|:---:|---|---
D1 | [Download (3.7G)](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/vid2vid/vid2vid_oleks.pt) | [Download (271M)](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/nerf/nerf_oleks.pt) | D2 | [Download (3.7G)](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/vid2vid/vid2vid_vlad.pt) | [Download (271M)](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/nerf/nerf_vlad.pt)
S1 | [Download (3.7G)](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/vid2vid/vid2vid_marc.pt) | [Download (271M)](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/nerf/nerf_marc.pt) | S2 | [Download (3.7G)](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/vid2vid/vid2vid_lan.pt) | [Download (271M)](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/nerf/nerf_lan.pt)
N1 | [Download (3.7G)](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/vid2vid/vid2vid_n1.pt) | [Download (271M)](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/nerf/nerf_n1.pt) | N2 | [Download (3.7G)](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/vid2vid/vid2vid_n2.pt) | [Download (271M)](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/nerf/nerf_n2.pt)
N3 | [Download (3.7G)](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/vid2vid/vid2vid_n3.pt) | [Download (271M)](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/nerf/nerf_n3.pt) | N4 | [Download (3.7G)](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/vid2vid/vid2vid_n4.pt) | [Download (271M)](https://dl.fbaipublicfiles.com/nsvf/neural_actor/models/nerf/nerf_n4.pt)

You can download the pretrained models for texture predictor and neural renderer of each character for [rendering pipeline](/docs/rendering_pipeline.md).

## Free Viewpoint Rendering Pipeline

We provide an [example](/docs/rendering_pipeline.md) of free-view video synthesis of a pre-trained human actor given arbitrary pose control.

## Train a new model

We provide an [example](/docs/training.md) for training the texture prediction and free-view neural rendering.

<!-- ## Train a new model

Given the dataset of a single scene (``{DATASET}``), we use the following command for training an NSVF model to synthesize novel views at ``800x800`` pixels, with a batch size of ``4`` images per GPU and ``2048`` rays per image. By default, the code will automatically detect all available GPUs.

In the following example, we use a pre-defined architecture ``nsvf_base`` with specific arguments:

* By setting ``--no-sampling-at-reader``, the model only samples pixels in the projected image region of sparse voxels for training.
* By default, we set the ray-marching step size to be the ratio ``1/8 (0.125)`` of the voxel size which is typically described in the ``bbox.txt`` file.
* It is optional to turn on ``--use-octree``. It will build a sparse voxel octree to speed-up the ray-voxel intersection especially when the number of voxels is larger than ``10000``.
* By setting ``--pruning-every-steps`` as ``2500``, the model performs self-pruning at every ``2500`` steps.
* By setting ``--half-voxel-size-at`` and ``--reduce-step-size-at`` as ``5000,25000,75000``,  the voxel size and step size are halved at ``5k``, ``25k`` and ``75k``, respectively.

Note that, although above parameter settings are used for most of the experiments in the paper, it is possible to tune these parameters to achieve better quality. Besides the above parameters, other parameters can also use default settings.

Besides the architecture ``nsvf_base``, you may check other architectures or define your own architectures in the file ``fairnr/models/nsvf.py``.

```bash
python -u train.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "0..100" --view-resolution "800x800" \
    --max-sentences 1 --view-per-batch 4 --pixel-per-view 2048 \
    --no-preload \
    --sampling-on-mask 1.0 --no-sampling-at-reader \
    --valid-views "100..200" --valid-view-resolution "400x400" \
    --valid-view-per-batch 1 \
    --transparent-background "1.0,1.0,1.0" --background-stop-gradient \
    --arch nsvf_base \
    --initial-boundingbox ${DATASET}/bbox.txt \
    --use-octree \
    --raymarching-stepsize-ratio 0.125 \
    --discrete-regularization \
    --color-weight 128.0 --alpha-weight 1.0 \
    --optimizer "adam" --adam-betas "(0.9, 0.999)" \
    --lr 0.001 --lr-scheduler "polynomial_decay" --total-num-update 150000 \
    --criterion "srn_loss" --clip-norm 0.0 \
    --num-workers 0 \
    --seed 2 \
    --save-interval-updates 500 --max-update 150000 \
    --virtual-epoch-steps 5000 --save-interval 1 \
    --half-voxel-size-at  "5000,25000,75000" \
    --reduce-step-size-at "5000,25000,75000" \
    --pruning-every-steps 2500 \
    --keep-interval-updates 5 --keep-last-epochs 5 \
    --log-format simple --log-interval 1 \
    --save-dir ${SAVE} \
    --tensorboard-logdir ${SAVE}/tensorboard \
    | tee -a $SAVE/train.log
```

The checkpoints are saved in ``{SAVE}``. You can launch tensorboard to check training progress:

```bash
tensorboard --logdir=${SAVE}/tensorboard --port=10000
```

There are more examples of training scripts to reproduce the results of our paper under [examples](./examples/train/).

## Evaluation

Once the model is trained, the following command is used to evaluate rendering quality on the test views given the ``{MODEL_PATH}``.

```bash
python validate.py ${DATASET} \
    --user-dir fairnr \
    --valid-views "200..400" \
    --valid-view-resolution "800x800" \
    --no-preload \
    --task single_object_rendering \
    --max-sentences 1 \
    --valid-view-per-batch 1 \
    --path ${MODEL_PATH} \
    --model-overrides '{"chunk_size":512,"raymarching_tolerance":0.01,"tensorboard_logdir":"","eval_lpips":True}' \
```

Note that we override the ``raymarching_tolerance`` to ``0.01`` to enable early termination for rendering speed-up.

## Free Viewpoint Rendering

Free-viewpoint rendering can be achieved once a model is trained and a rendering trajectory is specified. For example, the following command is for rendering with a circle trajectory (angular speed 3 degree/frame, 15 frames per GPU). This outputs per-view rendered images and merge the images into a ``.mp4`` video in ``${SAVE}/output`` as follows:

<img src='docs/figs/results.gif'/>

By default, the code can detect all available GPUs.

```bash
python render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --model-overrides '{"chunk_size":512,"raymarching_tolerance":0.01}' \
    --render-beam 1 --render-angular-speed 3 --render-num-frames 15 \
    --render-save-fps 24 \
    --render-resolution "800x800" \
    --render-path-style "circle" \
    --render-path-args "{'radius': 3, 'h': 2, 'axis': 'z', 't0': -2, 'r':-1}" \
    --render-output ${SAVE}/output \
    --render-output-types "color" "depth" "voxel" "normal" --render-combine-output \
    --log-format "simple"
```

Our code also supports rendering for given camera poses.
For instance, the following command is for rendering with the camera poses defined in the 200-399th files under folder ``${DATASET}/pose``:

```bash
python render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --model-overrides '{"chunk_size":512,"raymarching_tolerance":0.01}' \
    --render-save-fps 24 \
    --render-resolution "800x800" \
    --render-camera-poses ${DATASET}/pose \
    --render-views "200..400" \
    --render-output ${SAVE}/output \
    --render-output-types "color" "depth" "voxel" "normal" --render-combine-output \
    --log-format "simple"
```

The code also supports rendering with camera poses defined in a ``.txt`` file. Please refer to this [example](./examples/render/render_jade.sh).

## Extract the Geometry

We also support running marching cubes to extract the iso-surfaces as triangle meshes from a trained NSVF model and saved as ``{SAVE}/{NAME}.ply``. 
```bash
python extract.py \
    --user-dir fairnr \
    --path ${MODEL_PATH} \
    --output ${SAVE} \
    --name ${NAME} \
    --format 'mc_mesh' \
    --mc-threshold 0.5 \
    --mc-num-samples-per-halfvoxel 5
```
It is also possible to export the learned sparse voxels by setting ``--format 'voxel_mesh'``.
The output ``.ply`` file can be opened with any 3D viewers such as [MeshLab](https://www.meshlab.net/). 

<img src='docs/figs/snapshot_meshlab.png'/> -->

## License

NeuralActor is under [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license.
The license applies to the pre-trained models as well.

## Citation

Please cite as 
```bibtex
@article{liu2021neural,
      title={Neural Actor: Neural Free-view Synthesis of Human Actors with Pose Control}, 
      author={Lingjie Liu and Marc Habermann and Viktor Rudnev and Kripasindhu Sarkar and Jiatao Gu and Christian Theobalt},
      year={2021},
      journal = {ACM Trans. Graph.(ACM SIGGRAPH Asia)}
}
```

If you use our dataset, please note that D1 and D2 are orginally from the paper [Real-time Deep Dynamic Characters](https://people.mpi-inf.mpg.de/~mhaberma/projects/2021-ddc/), and S1 and S2 are orginally from the paper [DeepCap: Monocular Human Performance Capture Using Weak Supervision](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/). We process these four datasets in our Neural Actor format. Therefore, please also consider citing the following references:

```bibtex
@article{habermann2021,
	author = {Marc Habermann and Lingjie Liu and Weipeng Xu and Michael Zollhoefer and Gerard Pons-Moll and Christian Theobalt},
	title = {Real-time Deep Dynamic Characters},
	journal = {ACM Transactions on Graphics}, 
	month = {aug},
	volume = {40},
	number = {4}, 
	articleno = {94},
	year = {2021}, 
	publisher = {ACM}
} 

@inproceedings{deepcap,
    title = {DeepCap: Monocular Human Performance Capture Using Weak Supervision},
    author = {Habermann, Marc and Xu, Weipeng and Zollhoefer, Michael and Pons-Moll, Gerard and Theobalt, Christian},
    booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {jun},
    organization = {{IEEE}},
    year = {2020},
}
```