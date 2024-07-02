# Knowledge Transfer with Simulated Inter-Image Erasing for Weakly Supervised Semantic Segmentation


Network Architecture
--------------------
The architecture of our proposed approach is as follows
![network](framework.png)



## Prerequisite
* Tested on Ubuntu 18.04, with Python 3.8, PyTorch 1.8.2, CUDA 11.3.

* You can create conda environment with the provided yaml file.
```
conda env create -f wsss_new.yaml
```
* Download [The PASCAL VOC 2012 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/):
and put it under ./data/ folder.

### Test
* Download our pretrained weight [039net_main.pth](https://ktse.oss-cn-shanghai.aliyuncs.com/039net_main.pth) (PASCAL, seed: 67% mIoU) and put it under ./experiments/ktse1/ckpt/ folder.
```
python infer.py --name ktse1 --model ktse --load_epo 39 --dict  --infer_list voc12/train_aug.txt
```
```
python evaluation.py --name ktse1 --task cam --dict_dir dict
```


### Training
* Download the initial weights pretrained on Imagenet [ilsvrc-cls_rna-a1_cls1000_ep-0001.params](https://ktse.oss-cn-shanghai.aliyuncs.com/ilsvrc-cls_rna-a1_cls1000_ep-0001.params) and put it under ./pretrained/ folder.
* Please specify the name of your experiment (e.g., ktse1).
```
python train.py --name ktse1 --model ktse
```




