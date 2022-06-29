export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=

DATA="DeepcapLan"
RES="1024x1024"
VALID="0,9,15,19"

DATASET=/private/home/jgu/data/shapenet/DEBUG/${DATA}/predicted_testing
CANONICAL=${DATASET}/canonical.obj
OBJUV=${DATASET}/cut_alongsmplseams.obj
WEIGHT=${DATASET}/skinning_weight.txt
CENTER=$(python tools/get_centers.py ${DATASET}/transform)
echo "get center" $CENTER

# generate texture map


# generate video with test sequence, novel camera poses
mkdir -p results/test_video/lan


python render.py ${DATASET}/vid2vid_t_latest \
        --start-end "(3000,6000)" \
        --task single_object_rendering \
        --path checkpoints/neuralactor_lan.pt \
        --batch-size 1 \
        --load-video-dataset \
        --render-beam 1 \
        --render-angular-speed 2 \
        --render-save-fps 40 \
        --render-num-frames 1 \
        --render-at-vector $CENTER \
        --render-up-vector "(0,-1,0)" \
        --render-path-args "{'radius': 2, 'axis': 'y', 'h': 0.3, 't0': -1.5707963267948966}" \
        --render-resolution ${RES} \
        --render-output results/test_video/lan \
        --render-output-types "color" "normal" "voxel" \
        --render-combine-output \
        --render-maximum-frames 80
