% script_fusionOthers.m
% -------------------------------------------------------------------
% 
% Date:    10/04/2013
% Last modified: 20/04/2015
% -------------------------------------------------------------------

function Fusion_test_for_misregistered_img()

%     clear
    close all
    clc

    %% ------ Input the images ----------------
     % -------------- The color -----------------  
%     path1 = '.\image\mis-registered-images\temple_A.bmp';
%     path2 = '.\image\mis-registered-images\temple_B.bmp';
     path1 = 'image1_re.png';
     path2 = 'image2_re.png';
    % ------------- The Gray ----------------
%     path1 = '.\image\mis-registered-images\lab_A.tif';
%     path2 = '.\image\mis-registered-images\lab_B.tif';
%     path1 = '.\image\mis-registered-images\clock_A.bmp';
%     path2 = '.\image\mis-registered-images\clock_B.bmp';
    % -----------------------------------------
    
    [img1, img2] = PickName(path1, path2, 0);
    paraShow.fig = 'Input 1';
    paraShow.title = 'Org1';
    ShowImageGrad(img1, paraShow)
    paraShow.fig = 'Input 2';
    paraShow.title = 'Org2';
    ShowImageGrad(img2, paraShow)
    %% ---- The parameters -----
    % ----------- the multi scale -----
    para.Scale.lsigma = 4;
    para.Scale.ssigma = 0.5;
    para.Scale.alpha = 0.5;
    % -------------- the Merge parameter for fusion of mis-registered images -------------
    %para.Merge.per = 0.1;
    para.Merge.per = 0.01;
    para.Merge.margin = 4*para.Scale.lsigma;
    para.Merge.method = 2;
    % ------------- the Reconstruct parameter -----------
    para.Rec.iter = 500;
    para.Rec.res = 1e-6;
    para.Rec.modify = 5;
    para.Rec.iniMode = 'weight';   
    
    %% ---- MWGF implementation ------
    imgRec = MWGFusion(img1, img2, para);

    % --- Show the result ------
    paraShow.fig = 'fusion result';
    paraShow.title = 'MWGF';
    ShowImageGrad(imgRec, paraShow);
    imwrite(uint8(imgRec), 'result.jpg', 'jpeg');
    
   
end
