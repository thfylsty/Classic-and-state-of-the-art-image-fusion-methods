# DDcGAN-tensorflow:<br> infrared and visible image fusion via dual-discriminator conditional generative adversarial network
This work can be applied for<br> 
1) multi-resolution infrard and visible image fusion<br>
2) same-resolution infrared and visible image fusion<br>
2) PET and MRI image fusion<br>  

## Framework:
<div align=center><img src="https://github.com/hanna-xu/DDcGAN/blob/master/figures/framework.png" width="600" height="280"/></div><br>

## Generator architecture:
<div align=center><img src="https://github.com/hanna-xu/DDcGAN/blob/master/figures/Generator.png" width="520" height="280"/></div><br>

## Training dataset:
1) [vis-ir dataset](https://pan.baidu.com/s/1xKF9GBjZ92uhYhZ5gk5vLg) (password:nh2r).<br>
2) [PET-MRI dataset](https://pan.baidu.com/s/1sOPrLmVKG6fgNGP-T_2bXQ) (password: 5d9y).<br>
The code to create your own training dataset can be found [*here*](https://github.com/hanna-xu/utils).

If this work is helpful to you, please cite it as: 
```
@article{ma2020ddcgan,
  title={DDcGAN: A Dual-Discriminator Conditional Generative Adversarial Network for Multi-Resolution Image Fusion},
  author={Ma, Jiayi and Xu, Han and Jiang, Junjun and Mei, Xiaoguang and Zhang, Xiao-Ping},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={4980--4995},
  year={2020},
  publisher={IEEE}
}s
```

The previous version of our work can be seen in this paper:<br>
```
@inproceedings{xu2019learning,
  title={Learning a generative model for fusing infrared and visible images via conditional generative adversarial network with dual discriminators},
  author={Xu, Han and Liang, Pengwei and Yu, Wei and Jiang, Junjun and Ma, Jiayi},
  booktitle={proceedings of Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19)},
  pages={3954--3960},
  year={2019}
}
```
This code is base on the code of [*DenseFuse*](https://github.com/hli1221/imagefusion_densefuse).

If you have any question, please email to me (xu_han@whu.edu.cn).
