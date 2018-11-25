NETWORK=trained_models/resnet101/eccv.pth.tar

python eval.py --feature-extraction-cnn resnet101 --model $NETWORK --eval-dataset pf_pascal --eval-dataset-path datasets/proposal-flow-pascal --csv-path datasets/proposal-flow-pascal --gpu 0 --pck-alpha 0.1
