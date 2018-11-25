# Deep Semantic Matching with Foreground Detection and Cycle-Consistency

Pytorch implementation of our method for weakly supervised semantic matching.

Contact: Yun-Chun Chen (ycchen918 at citi dot sinica dot edu dot tw)

Please cite our paper if you find it useful for your research.

```
@inproceedings{Chen_WeakMatchNet_2018,
  author = {Y.-C. Chen and P.-H. Huang and L.-Y. Yu and J.-B. Huang and M.-H. Yang and Y.-Y. Lin},
  booktitle = {Asian Conference on Computer Vision (ACCV)},
  title = {Deep Semantic Matching with Foreground Detection and Cycle-Consistency},
  year = {2018}
}
```

## Installation
* Install PyTorch

* Clone this repo
```
git clone https://github.com/YunChunChen/WeakMatchNet
cd WeakMatchNet
```
## Dataset
* Please use the [code](https://github.com/ignacio-rocco/weakalign/blob/master/data/download_datasets.py) to download the datasets and put it under the `data/` folder.


* Please download the pre-trained model for training [here](http://www.di.ens.fr/willow/research/weakalign/trained_models/weakalign_resnet101_affine_tps.pth.tar) and put it under the `trained_models/resnet101/` folder.


* Evaluation command

```
sh eval.sh
```

* Training command

```
sh train.sh
```

## Related Implementation and Dataset
* Rocco et al. Convolutional Neural Network Architecture for Geometric Matching. In CVPR, 2017. [[project]](https://www.di.ens.fr/willow/research/cnngeometric/) [[paper]](https://arxiv.org/pdf/1703.05593.pdf) [[code]](https://github.com/ignacio-rocco/cnngeometric_pytorch)
* Rocco et al. End-to-End Weakly-Supervised Semantic Alignment. In CVPR 2018. [[project]](https://www.di.ens.fr/willow/research/weakalign/) [[paper]](https://arxiv.org/pdf/1712.06861.pdf) [[code]](https://github.com/ignacio-rocco/weakalign)

## Acknowledgment
This code is heavily borrowed from [weakalign](https://github.com/ignacio-rocco/weakalign).

## Note
The model and code are available for non-commercial research purposes only.
* Nov 2018: code released!
