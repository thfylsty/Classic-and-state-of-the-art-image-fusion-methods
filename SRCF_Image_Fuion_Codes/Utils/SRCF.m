function [fusedImage,fusionMap] = SRCF(imgA0, imgB0, Dict)
%-------------------------------------------------------------------------
% Sparse Representation-based Classification for Fusion (SRCF)
%-------------------------------------------------------------------------
D = Dict.D; % learned dictionary
bb = sqrt(size(D,1)/2); % block size
imgSize = [size(imgA0,1),size(imgA0,2)];

if (size(imgA0,3)==3), imgA = rgb2gray(imgA0); else imgA = imgA0; end
if (size(imgB0,3)==3), imgB = rgb2gray(imgB0); else imgB = imgB0; end
imgA = double(imgA);
imgB = double(imgB);


%=====< Calculating Focus Measure Maps >=====%
FmapA = calcFocusMeasure(imgA, 5, 'EOL');
FmapB = calcFocusMeasure(imgB, 5, 'EOL');
FmapA = padarray(FmapA,[(bb-1)/2,(bb-1)/2],'symmetric');
FmapB = padarray(FmapB,[(bb-1)/2,(bb-1)/2],'symmetric');
FmapA(FmapA<1e-8)=0;
FmapB(FmapB<1e-8)=0;           
     

%=====<  Block Extraction from Focus Measure Maps >=====%
[blocksA,~] = my_im2col(FmapA(:,:,1),[bb,bb],1);
[blocksB,~] = my_im2col(FmapB(:,:,1),[bb,bb],1);
blocks = [blocksA;blocksB];
clear blocksA blocksB FmapA FmapB


%=====< Normalization >=====%
for jj = 1:20000:size(blocks,2)
    jumpSize = min(jj+20000-1,size(blocks,2));
    blocks(:,jj:jumpSize) = blocks(:,jj:jumpSize)./ repmat(sqrt(sum(blocks(:,jj:jumpSize).^2 )),[size(blocks,1),1]);
end
if sum(sum(isnan(blocks)))>0
    blocks(isnan(blocks)) = 0;
end
 

%=====< Sparse Coding of Focus Features >=====%
for jj = 1:20000:size(blocks,2)
    jumpSize = min(jj+20000-1,size(blocks,2));
    Coefs(:,jj:jumpSize) = omp(D'*double(blocks(:,jj:jumpSize)) ,D'*D, 5);
end


%=====< Sparse Representation-based Classification >=====%
classes = [0,1]; 
npts = size(blocks,2);
labelImage = zeros(imgSize);
scoreValue = zeros(imgSize);

classCorr = Dict.coefHist' * abs(Coefs); 
[maxVal,maxLoc] = max(classCorr,[],1);
labelImage(1:npts) = classes(maxLoc);
labelImage(isnan(maxVal))=0;
maxVal(isnan(maxVal)) = 0;
maxVal(maxVal<0) = 0;
scoreValue(1:npts) =  maxVal.*labelImage(1:npts)+maxVal.*(labelImage(1:npts)-1);
clear blocks Coefs classCorr


%=====< Fusion Map Regularization Using Graph-Cut Optimization >=====%
param.lambda = 15; % weight of smoothness term
param.sigma = 5; % parameter of smoothness energy function
fusionMap = FusionMapRegularize(labelImage, scoreValue, imgA, imgB, param);


%=====< Calculating Fused Image >=====%
fusedImage = zeros(size(imgA0));
if size(imgA0,3)==1
    fusedImage(fusionMap==1) = imgA0(fusionMap==1);
    fusedImage(fusionMap==0) = imgB0(fusionMap==0);
else
    for i = 1:size(imgA0,3)
        tmp1 = double(imgA0(:,:,i));
        tmp2 = double(imgB0(:,:,i));
        fusedImage(:,:,i) = tmp1.*fusionMap + tmp2.*(1-fusionMap);
    end
end
fusedImage = uint8(fusedImage);
return
