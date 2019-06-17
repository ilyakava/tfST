
# tfST

## tensorflow implementation of the scattering transform

This implementation example runs on Hyperspectral Data.

Please cite:

```
@article{tfST,
	title={Three-Dimensional Fourier Scattering Transform and Classification of Hyperspectral Images},
	author={Ilya Kavalerov and Weilin Li and Wojciech Czaja and Rama Chellappa},
	journal={arXiv preprint arXiv:TBA},
	year={2019}
}
```


### Usage

### Downloading Data

See [the GIC website](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) and download for example the "corrected Indian Pines" and "Indian Pines groundtruth" datasets.

### Running Classification

```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset=IP --data_root=/scratch0/ilya/locDoc/data/hyperspec/datasets/ --train_test_splits=Indian_pines_gt_traintest.mat
```

Will run classification on a training set size of 10\% with OA 98.30\%.

#### Create Custom Training/Testing Splits

One training/testing split is included. Create more by editing the variables `OUT_PATH`, `DATASET_PATH`, `ntrials`, and `datasetsamples` in `create_training_splits.m`, and running:

```
matlab -nodesktop -nosplash -r "create_training_splits"
```

## Versioning

Tested on Python 2.7.14 (Anaconda), tensorflow 1.10.1, cuda 9.0.176, cudnn-8.0. Red Hat Enterprise Linux Workstation release 7.6 (Maipo). GeForce GTX TITAN X.

## License

MIT

