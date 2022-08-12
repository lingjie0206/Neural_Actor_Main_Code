#! /bin/bash
#SBATCH -p gpu20
#SBATCH -c 4
#SBATCH --gres=gpu:4
#SBATCH -t 48:00:00
#SBATCH --job-name o1
#SBATCH -o /HPS/HumanBodyRetargeting7/work/slurm_logs/train-%A.out

# just for debugging
DATA="lan"
RES="1024x1024"
DATASET=/HPS/HumanBodyRetargeting2/work/Release_data_example/${DATA}/training_images

MODEL="joint_nerf_coarse"
SUFFIX="debug"
ARCH=${DATA}_${MODEL}_${SUFFIX}
SAVE=/HPS/HumanBodyRetargeting2/work/Release_data_example/checkpoints
mkdir -p $SAVE/$ARCH

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=

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

python train.py ${DATASET} --arch $MODEL --seed 2 \
    $COMMON_FLAGS $DATA_FLAGS $INPUT_FLAGS $ENCODER_FLAGS $NERF_FLAGS $LOSS_FLAGS $OPTIMIZER_FLAGS $TRAIN_FLAGS $CHECKPOINT_FLAGS $LOG_FLAGS