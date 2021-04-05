function [B_detail, B_smooth] = train_choose_detail(training_path, unit, col)
% choosing image patches which contain more detail

fileFolder=fullfile(training_path);
dirOutput=dir(fullfile(fileFolder,'*'));
num = length(dirOutput);
% fileNames={dirOutput.name}'; % all names
B = [];
for i = 3:num
    img = imread([training_path,dirOutput(i).name]);
    if size(img,3)>1
        img = rgb2gray(img);
    end

    [t1,t2] = size(img);
    img = img(1:floor(t1/unit)*unit, 1:floor(t2/unit)*unit);

    img = im2double(img);
    b = im2col(img, [unit, unit], 'distinct');
    B = cat(2, B, b);
end
% calculating SD valuee of each column B
B_mean = mean(B,1);
B_mean = repmat(B_mean, unit*unit, 1);
B_sd = sqrt(sum((B - B_mean).*(B - B_mean),1));
% choose >0.5
B_sd(find(B_sd<=0.5)) = 0;
B_sd(find(B_sd>0.5)) = 1;
B_mask = repmat(B_sd, unit*unit, 1);
B_smooth = B.*(1-B_mask);
B_detail = B.*B_mask;
B_smooth(:,all(B_smooth==0,1))=[];
B_detail(:,all(B_detail==0,1))=[];

[m1,n1] = size(B_detail);
[m2,n2] = size(B_smooth);

end