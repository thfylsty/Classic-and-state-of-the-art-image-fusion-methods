%=========================================================================
% Sparse Representation-based Classification for Fusion (SRCF) , v1.0
%
% M. Nejati, S. Samavi, S. Shirani, "Multi-focus Image Fusion Using 
% Dictionary-Based Sparse Representation", Information Fusion, vol. 25,
% Sept. 2015, pp. 72-84. 
%
% Please refer to the above paper if you use this software.
%=========================================================================
clc,clear all,close all;
addpath('Utils','Utils\ompbox10','Utils\Metrics','Utils\GCO\bin','Utils\GCO\');


%====< Test Dataset >====%
dataPath = 'Data\GrayscaleDataset';
% dataPath = 'Data\LytroDataset';
[dataset,Dict] = loadData(dataPath);


%====< Fusion of Mutifocus Image Pairs >====%
res = struct([]);
for i = 1:dataset.numImage

    fprintf('>> start of fusion for image (%d)...\n',i);

    %----< load source image pairs >----%
    imgA0 = imread(fullfile(dataset.dataPath, dataset.imagesA{i}));
    imgB0 = imread(fullfile(dataset.dataPath, dataset.imagesB{i}));
    if strcmp(dataset.imgSet,'Grayscale')
    if (size(imgA0,3)>1), imgA0 = rgb2gray(imgA0); end
    if (size(imgB0,3)>1), imgB0 = rgb2gray(imgB0); end
    end
    
    %---< Fusion >---%
    imgF0 = SRCF(imgA0, imgB0, Dict);
    
    if (size(imgA0,3)>1), imgA = rgb2gray(imgA0); else imgA = imgA0; end
    if (size(imgB0,3)>1), imgB = rgb2gray(imgB0); else imgB = imgB0; end
    if (size(imgF0,3)>1), imgF = rgb2gray(imgF0); else imgF = imgF0; end

    %=======================
    % Objective Evaluation
    %=======================
    MI = fusionMI(imgA,imgB,imgF); % no matter double or uint8
    NMI = fusionECC(imgA,imgB,imgF); % no matter double or uint8
    QABF = Qabf_eval(imgA,imgB,imgF); % no matter double or uint8
    VIFF = VIFF_Public(imgA,imgB,imgF); % no matter double or uint8
    fprintf('-------------------------------------------------------\n')
    fprintf(' %s: QABF = %.4f , VIFF = %.4f , NMI = %.4f\n',dataset.imagesA{i}(1:end-4),QABF,VIFF,NMI);
    fprintf('-------------------------------------------------------\n')
    res(i).MI = MI;
    res(i).NMI = NMI;
    res(i).QABF = QABF;
    res(i).VIFF = VIFF;
    
    %---< Display >---%
%     figure,imshow(imgA0,[]),title('Source Image 1')
%     figure,imshow(imgB0,[]),title('Source Image 2')
    figure,imshow(imgF0,[]),title('Fused Image')
    pause(2);
    close all;

end

