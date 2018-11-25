MODEL=resnet101
MODEL_DIR=trained_models/$MODEL
PRETRAINED_MODEL=$MODEL_DIR/weakalign.pth.tar

W_INLIER=0.0
W_CONSIS=0.0
W_TRANS=0.0

INLIER_LOSS=True
CONSISTENCY_LOSS=True
TRANSITIVITY_LOSS=True

GPU=0
EPOCH=30
BATCH_SIZE=6
LEARNING_RATE=5e-8

python train.py --feature-extraction-cnn $MODEL --model $PRETRAINED_MODEL --num-epochs $EPOCH --lr $LEARNING_RATE --gpu $GPU --inlier-loss $INLIER_LOSS --consistency-loss $CONSISTENCY_LOSS --transitivity-loss $TRANSITIVITY_LOSS --result-model-dir $MODEL_DIR --batch-size $BATCH_SIZE --w-inlier $W_INLIER --w-consis $W_CONSIS --w-trans $W_TRANS --bi-directional True
