set -e
exeFunc(){
    num_seq=$1
    sequence_length=$2
    python utils/lidar2voxel.py \
    --dataset ./mobilestereonet/lidar/ \
    --output ./kitti/dataset \
    --num_seq $num_seq \
    --sequence_length $sequence_length
}

sequence_length=10
for i in {00..21}
do
    exeFunc $i $sequence_length
done
