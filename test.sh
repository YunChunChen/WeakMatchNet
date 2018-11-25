MODEL=resnet101
MODEL_DIR=trained_models/$MODEL

W_INLIER=$2
W_COORD=0.0
W_TRANS=0.0

INLIER_LOSS=True
COORD_LOSS=False
TRANS_LOSS=False

GPU=$1
EPOCH=1
BATCH_SIZE=6
LEARNING_RATE=5e-8

PRETRAINED_MODEL=$MODEL_DIR/best_inlier_${W_INLIER}_coord_${W_COORD}_trans_${W_TRANS}.pth.tar

python test.py --feature-extraction-cnn $MODEL --model $PRETRAINED_MODEL --training-dataset pf-pascal --dataset-csv-path pf_pascal_data --dataset-image-path datasets/proposal-flow-pascal/ --eval-dataset-path datasets/proposal-flow-pascal --num-epochs $EPOCH --lr $LEARNING_RATE --gpu $GPU --inlier-loss $INLIER_LOSS --coord-loss $COORD_LOSS --transitivity-loss $TRANS_LOSS --result-model-dir $MODEL_DIR --batch-size $BATCH_SIZE --w-inlier $W_INLIER --w-coord $W_COORD --w-trans $W_TRANS --bi-directional True
