% MWGFFusion.m
% -------------------------------------------------------------------
% 
% Date:    27/04/2013
% Last modified: 29/10/2013
% -------------------------------------------------------------------
function imgRec = MWGFusion(img1, img2, para)
    
    %% ----------------- Compute the weights ------------------
    if size(img1, 3) == 1,
        img1Gray = img1;
        img2Gray = img2;
    else 
        img1Gray = RGBTOGRAY(img1);
        img2Gray = RGBTOGRAY(img2);
    end
    disp('Start MWGF algorithm ...')
    % ----- Compute the gradient ------
    [dx1, dy1] = GradientMethod(img1Gray, 'zhou'); 
    [dx2, dy2] = GradientMethod(img2Gray, 'zhou');
    dxdy1 = dx1+1i*dy1;
    dxdy2 = dx2+1i*dy2;
    
    para.Merge.oppo = 0;
    [~, ~, wt1, wt2] = WeightGradient(imresize(dxdy1, 0.25), imresize(dxdy2, 0.25), para);
    aa = wt1>wt2+eps;
    if sum(aa(:))/numel(wt1) > 0.5,
        para.Merge.oppo = 1;
    end
  
    % Compute the Large and small scale structure saliency Q
    para.LScale.sigma = para.Scale.lsigma;
    para.LScale.alpha = para.Scale.alpha;
    if isfield(para, 'LScale'),
        disp('Computing the large scale structure saliency Q ...')
        [~, ~, wtL1, wtL2] = WeightGradient(dxdy1, dxdy2, para.LScale); 
    end  
    
    para.SScale.sigma = para.Scale.ssigma;
    para.SScale.alpha = para.Scale.alpha;
    if isfield(para, 'SScale'),
        disp('Computing the small scale structure saliency Q ...')
        [~, ~, wtS1, wtS2] = WeightGradient(dxdy1, dxdy2, para.SScale); 
    else
        error('The scale must have the small scale');
    end
    
    % Combining multi-scale information 
    if exist('wtL1', 'var'),
        disp('Combining multi-scale information ...')
        wt1 = MergeWeights(wtL1, wtL2, wtS1, wtS2, para.Merge);
        ww1 = ordfilt2(wt1, 5, ones(3, 3));
        ww2 = 1 - ww1;
    else
        ww1 = wtS1 ./ (wtS1 + wtS2 +eps);
        ww2 = 1-ww1;
    end
    
    
    %% ----------------- Weighted gradient-based fusion --------------------
    imgRec = zeros(size(img1));
    disp('Reconstructing the fused image from the merged gradients ...')
    for ii = 1:size(img1, 3),
        [dx1, dy1] = GradientMethod(img1(:, :, ii), 'zhou'); 
        [dx2, dy2] = GradientMethod(img2(:, :, ii), 'zhou');
        
        dxdy1 = dx1+1i*dy1;
        dxdy2 = dx2+1i*dy2;
        dxdy = GradientMixWeightModify(dxdy1, dxdy2, ww1, ww2, para.Rec.modify);
        
        imgRec(:, :, ii) = RecByGraInitial(img1(:, :, ii), img2(:, :, ii), ww1, ww2, dxdy, para.Rec.iter, para.Rec.res, 0.1, para.Rec.iniMode);
    end
    
    %% 
    disp('End.')
    imgRec = min(imgRec, 255);
    imgRec = max(imgRec, 0);
    imgRec = round(imgRec);
end

