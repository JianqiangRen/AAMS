Attention-aware Multi-stroke Style Transfer
=====

This is the official [Tensorflow](https://www.tensorflow.org/) implementation of our paper:

Attention-aware Multi-stroke Style Transfer, CVPR 2019. [[Project]](https://sites.google.com/view/yuanyao/attention-aware-multi-stroke-style-transfer) [[arXiv]](https://arxiv.org/abs/1901.05127)

[Yuan Yao](mailto:yaoy92@gmail.com), [Jianqiang Ren](mailto:jianqiang.rjq@alibaba-inc.com), Xuansong Xie, Weidong Liu, [Yong-Jin Liu](https://cg.cs.tsinghua.edu.cn/people/~Yongjin/Yongjin.htm), [Jun Wang](http://www0.cs.ucl.ac.uk/staff/Jun.Wang/)

## Overview
This project provides an arbitrary style transfer method that achieves both faithful style transfer and visual consistency between the content and stylized images. The key idea of the proposed method is to employ self-attention mechanism, multi-scale style swap and a flexible stroke pattern fusion strategy to smoothly and adaptably apply suitable stroke patterns on different regions. In this manner, the synthesized images of our method can be more visually pleasing and generated in one feed-forward pass.
<div align='center'>
  <img src='https://github.com/JianqiangRen/AAMS/blob/master/images/guideline/motivation.jpg' height="350px">
</div>

## Examples
<div align='center'>
  <img src='https://github.com/JianqiangRen/AAMS/blob/master/images/guideline/fig1.jpg' height="850px">
</div>

## Prerequisites
- Python (version 2.7)
- Tensorflow (>=1.4)
- Numpy
- Matplotlib

## Download
* [MSCOCO](http://cocodataset.org/#home) dataset is applied for the training of the proposed self-attention autoencoder.
* Pre-trained [VGG-19](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz) model.

## Usage
### Test

Make sure there exists a sub-folder named test_result under images folder, then run 
```
$ python test.py --model tf_model/aams.pb \
                        --content images/content/lenna_cropped.jpg \
                        --style images/style/candy.jpg \
                        --inter_weight 1.0
```
both of the stylized image and the attention map will be generated in test_result.

Our model is trained with tensorflow 1.4.

### Train
Download the  [MSCOCO](http://cocodataset.org/#home) dataset and filter out images with unsuitable format(grayscale,etc) by running

```
$ python filter_training_images.py --dataset datasets/COCO_Datasets/val2014
```
then
```
$ python train.py --dataset datasets/COCO_Datasets/val2014
```
## Citation
If our work is useful for your research, please consider citing:

    @inproceedings{yao2019attention,
	    title={Attention-aware Multi-stroke Style Transfer},
	    author={Yao, Yuan and Ren, Jianqiang and Xie, Xuansong and Liu, Weidong and Liu, Yong-Jin and Wang, Jun},
	    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	    year={2019}
    }
    
## License
© Alibaba, 2019. For academic and non-commercial use only.

## Acknowledgement
We express gratitudes to the style-agnostic style transfer works including [Style-swap](https://arxiv.org/abs/1612.04337), [WCT](https://arxiv.org/abs/1705.08086) and [Avatar-Net](https://arxiv.org/abs/1805.03857), as we benefit a lot from both their papers and codes.

## Contact
If you have any questions or suggestions about this paper, feel free to contact [Yuan Yao](mailto:yaoy92@gmail.com) or [Jianqiang Ren](mailto:jianqiang.rjq@alibaba-inc.com).
