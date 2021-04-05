% PruneWeights.m
% -------------------------------------------------------------------
% Date:    17/04/2013
% Last modified: 29/10/2013
% -------------------------------------------------------------------

function [ww1, ww2] = MergeWeights(wtL1, wtL2, wtS1, wtS2, para)
    % ----------- Set the parameter -------------------
    show = 0;
    method = 1;
    per = 0.1;
    margin = ceil(4*4);
    basethres = 0.8;%0.8
    if exist('para', 'var') && isfield(para, 'show'),
        show = para.show;
    end
    if exist('para', 'var') && isfield(para, 'method'), 
        method = para.method;
    end
    if exist('para', 'var') && isfield(para, 'per'),
        per = para.per;
    end
    if exist('para', 'var') && isfield(para, 'margin'),
        margin = para.margin;
    end
    if exist('para', 'var') && isfield(para, 'basethres'),
        basethres = para.basethres;
    end
    % ----------- Define the unknown region and the definite focused and defocused region -------------
    if para.oppo,
        baseLL = wtL2 > wtL1+eps+basethres;
        tt = wtS1;
        wtS1 = wtS2;
        wtS2 = tt;
    else
        baseLL = wtL1 > wtL2+eps+basethres;
    end
    
    % --- fill the small gap and the hole ----
    se = strel('disk', ceil(9));
    baseLL = imclose(baseLL, se);
    
    weightPost = zeros(size(wtL1));
    LL = bwlabel(baseLL, 8);
    tempHist = tabulate(LL(:));
    [~, tempIdx] = max(tempHist(2:end, 2));
    weightPost(LL == tempIdx) = 1;

    % ----- Delete the hole --
    [L,num]=bwlabel(~weightPost); % select the max region 0
    maxarea = 0;
    maxindex =0;
    for i = 1:num,
        temp = length( find(L==i) );
        if (temp > maxarea),
            maxarea = temp;
            maxindex = i;
        end
    end
      
    bw = (L == maxindex); % select the max region 0
    
    weightPost(bw) = 0;
    weightPost(~bw) = 1;
    weightPost = DeleteLine(weightPost);
    % ------------------------
    clear tempHist tempIdx LL

    seErode = strel('square', ceil(margin));
    tempErode = imerode(weightPost, seErode);
    seDilate = strel('square', ceil(margin*1.4));
    tempDilate = imdilate(weightPost, seDilate);
    validIdx2 = tempDilate == 0;
    validIdx1 = tempErode ~= 0;
    validIdxMid = (tempDilate -tempErode) ~= 0;

    wt1 = zeros(size(wtL1));
    wt2 = zeros(size(wtL2));
    wt1(validIdx1) = 1;
    wt2(validIdx2) = 1;
    % -------------- Various of Method -------------
    switch method,    
        case 1, %
            alpha = 5;
            subSS = wtS1 - wtS2;
           
            tempIdx = and(subSS > 0, validIdx2);
            temp = sort(wtS1(tempIdx(:)), 'descend');
            thre1 = temp(ceil(length(temp) * per));

            validIdx = and(subSS >=0, validIdxMid);
            wt1(validIdx) = (1+exp(alpha*(wtS1(validIdx)-thre1)./(thre1+eps)))./ ...
                (1+exp(alpha*(wtS1(validIdx)-thre1)./(thre1+eps))+exp((wtS2(validIdx)-wtS1(validIdx))./(wtS2(validIdx)+wtS1(validIdx)+eps))); % adopt the soft-max
            wt2(validIdx) = 1 - wt1(validIdx); % adopt the soft-max

            validIdx = and(subSS <0, validIdxMid);
            wt2(validIdx) = 1./ ...
                (1+exp((-wtS2(validIdx)+wtS1(validIdx))./(wtS2(validIdx)+wtS1(validIdx)+eps))); % adopt the soft-max
            wt1(validIdx) = 1 - wt2(validIdx); % adopt the soft-max
            
            %---
            ww1 = wt1;
            ww2 = wt2;  
        case 2,
            alpha = 5;
            subSS = wtS1 - wtS2;
            tempIdx = and(subSS < 0, validIdx1);
            temp = sort(wtS2(tempIdx(:)), 'descend');
            thre2 = temp(ceil(length(temp) * per));

            tempIdx = and(subSS > 0, validIdx2);
            temp = sort(wtS1(tempIdx(:)), 'descend');
            thre1 = temp(ceil(length(temp) * per));

            validIdx = and(subSS >=0, validIdxMid);
            wt1(validIdx) = (1+exp(alpha*(wtS1(validIdx)-thre1)./(thre1+eps)))./ ...
                (1+exp(alpha*(wtS1(validIdx)-thre1)./(thre1+eps))+exp((wtS2(validIdx)-wtS1(validIdx))./(wtS2(validIdx)+wtS1(validIdx)+eps))); % adopt the soft-max
            wt2(validIdx) = 1 - wt1(validIdx); % adopt the soft-max

            validIdx = and(subSS <0, validIdxMid);
            wt2(validIdx) = (1+exp(alpha*(wtS2(validIdx)-thre2)./(thre2+eps)))./ ...
                (1+exp(alpha*(wtS2(validIdx)-thre2)./(thre2+eps))+exp((-wtS2(validIdx)+wtS1(validIdx))./(wtS2(validIdx)+wtS1(validIdx)+eps))); % adopt the soft-max
            wt1(validIdx) = 1 - wt2(validIdx); % adopt the soft-max
            
            %---
            ww1 = wt1;
            ww2 = wt2;
        case 3,
            wt1(validIdxMid) = wtS1(validIdxMid);
            wt2(validIdxMid) = wtS2(validIdxMid);
            % --------------------------
            
            ww1 = wt1 ./ (wt1 +wt2+eps);
            ww2 = wt2 ./ (wt1 +wt2+eps);
            

        otherwise,
            error('The method is wrong');
    end
    % ----------------------------------------------
    
    if para.oppo,
        tt = ww1;
        ww1 = ww2;
        ww2 = tt;
    end
    
    % ------------- Display the Weights ------------
    if show,
        tempWeights = zeros(size(weightPost));
        if para.oppo,
            tempWeights(validIdx2) = 255;    
        else
            tempWeights(validIdx1) = 255;    
        end
        tempWeights(validIdxMid) = 125;
        paraShow.fig = ['detected region'];
        paraShow.title = ['detected region'];
        ShowImageGrad(tempWeights, paraShow); % Show the Margin
        
    end    
    % ----------------------------------------------
end

function mm = DeleteLine(nn)
    for ii = 1:size(nn,2),      
        befind = find(nn(:, ii), 1, 'first');
        afind = find(~nn(:, ii), 1, 'first');
        if and(befind == 1, afind < 13),
            nn(befind:afind, ii) = 0;
        end
        
        befind = find(nn(:, ii), 1, 'last');
        afind = find(~nn(:, ii), 1, 'last');
        if and(befind == size(nn,1), befind-afind< 13),
            nn(afind:befind, ii) = 0;
        end 

    end
    
    for ii = 1:size(nn,1),        
        befind = find(nn(ii, :), 1, 'first');
        afind = find(~nn(ii, :), 1, 'first');
        if and(befind == 1, afind < 13),
            nn(ii, befind:afind) = 0;
        end
        
        befind = find(nn(ii, :), 1, 'last');
        afind = find(~nn(ii, :), 1, 'last');
        if and(befind == size(nn,2), befind-afind < 13),
            nn(ii, afind:befind) = 0;
        end 
    end
    
    mm = nn;
end
