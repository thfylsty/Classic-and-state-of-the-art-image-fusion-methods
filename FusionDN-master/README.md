# FusionDN<br>
This is the code of the following paper (tensorflow):<br>
*@inproceedings{xu2020aaai,<br>
  title={FusionDN: A Unified Densely Connected Network for Image Fusion},<br>
  author={Xu, Han and Ma, Jiayi and Le, Zhuliang and Jiang, Junjun and Guo, Xiaojie},<br>
  booktitle={proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence},<br>
  year={2020}<br>
}*<br>

It is a unified model for multiple image fusion tasks, including visible and infrared image fusion, multi-exposure image fusion and multi-focus image fusion.<br>


## Framework:<br>
 Overall procedure:<br>
<div align=center><img src="https://github.com/hanna-xu/FusionDN/blob/master/imgs/procedure.jpg" width="440" height="290"/></div><br>

Intuitive description of data flow and the process of EWC:<br>
<div align=center><img src="https://github.com/hanna-xu/FusionDN/blob/master/imgs/MultiTask.jpg" width="510" height="200"/></div><br>

## Fused results:<br>
<div align=center><img src="https://github.com/hanna-xu/FusionDN/blob/master/imgs/res1.jpg" width="900" height="490"/></div>
<div align=center><img src="https://github.com/hanna-xu/FusionDN/blob/master/imgs/res2.jpg" width="900" height="400"/></div>

## To train:<br>
CUDA_VISIBLE_DEVICES=0,1 python main.py (2 GPUs are needed)<br><br>

## To test:<br>
CUDA_VISIBLE_DEVICES=0 python test_main.py<br><br>

## Tips:<br>
Large files should be downloaded separately, including the following files: <br>
#### For training:<br>
* [Training dataset](https://pan.baidu.com/s/18hQyGNlhVb3oXAXOJ5LZCQ)<br>
* [vgg16](https://pan.baidu.com/s/1vK3l8rzgAkxcKpLvnFAwXA) (put it in 'VGGnet/')<br>

#### For testing:<br>
* [Trained model](https://pan.baidu.com/s/1zH6DC0N78Mw2jvPRB2I6HA)
