function [Dataset,Dict] = loadData(dataPath)

if ~isempty(strfind(dataPath,'GrayscaleDataset'))
    imgSet = 'Grayscale';
    load('Data\D_Gray.mat');
elseif ~isempty(strfind(dataPath,'LytroDataset'))
    imgSet = 'Lytro';
    load('Data\D_Lytro.mat');
else
    imgSet = 'Unknown';
    load('Data\D_Lytro.mat');
end

Dataset.imgSet = imgSet;
Dataset.dataPath = dataPath;

switch imgSet
    
    case 'Grayscale'
        img_dir = dir(fullfile(dataPath,'*_1.*'));
        img_num = length(img_dir);
        for i = 1:img_num
            imgName = img_dir(i).name;
            Dataset.imagesA{i,1} = imgName;
            Dataset.imagesB{i,1} = [imgName(1:end-5),'2.',imgName(end-2:end)];
        end
        Dataset.numImage = img_num;
        
    case 'Lytro'
        imgA_dir = dir(fullfile(dataPath,'*-A.jpg'));
        imgB_dir = dir(fullfile(dataPath,'*-B.jpg'));
        img_num = length(imgA_dir);
        for i = 1:img_num
            Dataset.imagesA{i,1} = imgA_dir(i).name;
            Dataset.imagesB{i,1} = imgB_dir(i).name;
        end
        Dataset.numImage = img_num;
        
    case 'Unknown'
        ext = {'*.jpg','*.png','*.bmp','*.tif','*.gif'};
        img_dir   = [];
        for e = 1:numel(ext)
            img_dir = [img_dir; dir(fullfile(dataPath,ext{e}))];
        end
        img_num = length(img_dir);
        j = 1;
        for i = 1:2:img_num
            Dataset.imagesA{j,1} = img_dir(i).name;
            Dataset.imagesB{j,1} = img_dir(i+1).name;
            j = j + 1;
        end
        Dataset.numImage = img_num/2;
end
return
