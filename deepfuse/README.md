# Deepfuse - Tensorflow

This code is based on [K. Ram Prabhakar, V Sai Srikar, R. Venkatesh Babu. DeepFuse: A Deep Unsupervised Approach for Exposure Fusion with Extreme
Exposure Image Pairs. ICCV2017, pp. 4714-4722](http://openaccess.thecvf.com/content_iccv_2017/html/Prabhakar_DeepFuse_A_Deep_ICCV_2017_paper.html)

![](https://github.com/hli1221/Imagefusion_deepfuse/blob/master/figure/framework.png)
In this code, for all conv layers, the filter size is 3\*3. And this code is not a complete version for DeepFuse, we just implement one channel fusion method which use CNN network. 

This code is not exactlly same with paper in ICCV2017. The aim of the training process is to reconstruct the input image by this network. The encoder(C1, C2) is used to extract image features and decoder(C3, C4, C5) is a reconstruct tool. The fusion strategy( Tensor addition) is only used in testing process. 

We train this network using [Microsoft COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)(T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 3-5.) as input images which contains 80000 images and all resize to 256×256 and RGB images are transformed to gray ones.

# Citation
```
@misc{li2017deepfuse,
    author = {Hui Li},
    title = {CODE: Image Fusion based on Deepfuse},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/hli1221/Imagefusion_deepfuse}}
  }
```
