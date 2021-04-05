# IFCNN
Project page of  "[IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network](https://authors.elsevier.com/a/1ZTXt5a7-GbZZX),  Information Fusion, 54 (2020) 99-118". 



### Requirements
- pytorch=0.4.1
- python=3.x
- torchvision
- numpy
- opencv-python
- jupyter notebook (optional)
- anaconda (suggeted)

### Configuration
```bash
# Create your virtual environment using anaconda
conda create -n IFCNN python=3.5

# Activate your virtual environment
conda activate IFCNN

# Install the required libraries
conda install pytorch=0.4.1 cuda80 -c pytorch
conda install torchvision numpy jupyter notebook
pip install opencv-python
```


### Usage
```bash
# Clone our code
git clone https://github.com/uzeful/IFCNN.git
cd IFCNN/Code

# Remember to activate your virtual enviroment before running our code
conda activate IFCNN

# Replicate our image method on fusing multiple types of images
python IFCNN_Main.py

# Or run code part by part in notebook
jupyter notebook IFCNN_Notebook.ipynb
```



### Typos
1. Eq. (4) in our paper is wrongly written, the correct expression can be referred to the official expression in [OpenCV document](https://docs.opencv.org/3.4.2/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa), i.e., <img src="https://latex.codecogs.com/gif.latex?G(i)=\alpha&space;\cdot&space;e^{-\frac{[i-(ksize-1)/2]^2}{2\sigma^2}}" title="G(i)=\alpha \cdot e^{-\frac{[i-(ksize-1)/2]^2}{2\sigma^2}}" />, where <img src="https://latex.codecogs.com/gif.latex?i=0&space;\cdots&space;(ksize-1)" title="i=0 \cdots (ksize-1)" />, <img src="https://latex.codecogs.com/gif.latex?ksize=2\times{kr}&plus;1" title="ksize=2\times{kr}+1" />, <img src="https://latex.codecogs.com/gif.latex?\sigma=0.6\times(ksize-1)&plus;0.8" title="\sigma=0.6\times(ksize-1)+0.8" />, and <img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /> is the scale factor chosen for achieving <img src="https://latex.codecogs.com/gif.latex?\sum&space;G\left(i\right)=1" title="\sum G\left(i\right)=1" />.
2. Stride and padding parameters of CONV4 are respectively 1 and 0, rather than both 0.



### Highlights
- Propose a general image fusion framework based on convolutional neural network
- Demonstrate good generalization ability for fusing various types of images
- Perform comparably or even better than other algorithms on four image datasets
- Create a large-scale and diverse multi-focus image dataset for training CNN models
- Incorporate perceptual loss to boost the model’s performance



### Architecture of our image fusion model
![flowchart](https://github.com/uzeful/IFCNN/blob/master/flowchart.png)



### Comparison Examples
1. Multi-focus image fusion
![CMF05](https://github.com/uzeful/IFCNN/blob/master/Comparisons/CMF05.png)


2. Infrared and visual image fusion
![CMF05](https://github.com/uzeful/IFCNN/blob/master/Comparisons/IVroad.png)


3. Multi-modal medical image fusion
![MDc02](https://github.com/uzeful/IFCNN/blob/master/Comparisons/MDc02.png)


4. Multi-exposure image fusion
![MEdoor](https://github.com/uzeful/IFCNN/blob/master/Comparisons/MEdoor.png)



### Other Results of Our Model
1. Multi-focus image dataset: [Results/CMF](https://github.com/uzeful/IFCNN/tree/master/Results/CMF)
2. Infrared and visual image dataset: [Results/IV](https://github.com/uzeful/IFCNN/tree/master/Results/IV)
3. Multi-modal medical image dataset: [Results/MD](https://github.com/uzeful/IFCNN/tree/master/Results/MDDataset)
4. Multi-exposure image dataset: [Results/ME](https://github.com/uzeful/IFCNN/tree/master/Results/ME)



### Citation
If you find this code is useful for your research, please consider to cite our paper. Yu Zhang, Yu Liu, Peng Sun, Han Yan, Xiaolin Zhao, Li Zhang, [IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network](https://authors.elsevier.com/a/1ZTXt5a7-GbZZX),  Information Fusion, 54 (2020) 99-118.

```
@article{zhang2020IFCNN,
  title={IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network},
  author={Zhang, Yu and Liu, Yu and Sun, Peng and Yan, Han and Zhao, Xiaolin and Zhang, Li},
  journal={Information Fusion},
  volume={54},
  pages={99--118},
  year={2020},
  publisher={Elsevier}
}
```
