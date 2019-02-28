# Attention-aware Multi-stroke Style Transfer

This is the offical Tensorflow implementation of [Attention-aware Multi-stroke Style Transfer](https://arxiv.org/abs/1901.05127), CVPR 2019

[Yuan Yao](mailto:yaoy92@gmail.com), [Jianqiang Ren](mailto:jianqiang.rjq@alibaba-inc.com), Xuansong Xie, [Weidong Liu](https://www.tsinghua.edu.cn/publish/csen/4623/2010/20101224001537675975573/20101224001537675975573_.html), [Yong-Jin Liu](http://media.cs.tsinghua.edu.cn/en/liuyj), Jun Wang

<div align='center'>
  <img src='https://github.com/JianqiangRen/AAMS/blob/master/images/guideline/motivation.jpg' height="350px">
</div>

## Requirement
- Python (version 2.7)
- Tensorflow (>=1.4)
- Numpy
- Matplotlib

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
```
$ python train.py --dataset datasets/COCO_Datasets/val2014
```

## Examples
<div align='center'>
  <img src='https://github.com/JianqiangRen/AAMS/blob/master/images/guideline/fig1.jpg' height="850px">
</div>



## Acknowledgement
we acknowledge [avatar-net](https://github.com/LucasSheng/avatar-net) for their work.
