% WeightGradient.m
% -------------------------------------------------------------------
% Date:    2/04/2013
% Last modified: 16/04/2015
% -------------------------------------------------------------------
function [ww1, ww2, wt1, wt2] = WeightGradient(dxdy1, dxdy2, para)
    
    weiType = 'eigdecxq';
    sigma = 6;
    alpha = 0.5;
    % -------- Check the parameter -------------
    if isfield(para, 'weight'),
        weiType = para.weight;
    end
    if isfield(para, 'sigma'),
        sigma = para.sigma;
    end
    if isfield(para, 'alpha'),
        alpha = para.alpha;
    end
    % -----------------------------------------
    if strcmp(weiType, 'eigdecxq'),
%         disp('The weight will be equal eig decompose and the sqrt(s^2)');
        [~, cc1] = EigDecBlock(dxdy1, sigma);
        [~, cc2] = EigDecBlock(dxdy2, sigma);
        
        wt1 = sqrt((sqrt(cc1(1,:,:))+sqrt(cc1(2,:,:))).^2 + alpha*(sqrt(cc1(1,:,:))-sqrt(cc1(2,:,:))).^2);
        wt1 = squeeze(wt1);
        wt2 = sqrt((sqrt(cc2(1,:,:))+sqrt(cc2(2,:,:))).^2 + alpha*(sqrt(cc2(1,:,:))-sqrt(cc2(2,:,:))).^2);
        wt2 = squeeze(wt2);
        
        % normlaization
        ww1 = sqrt(wt1 ./ (wt1 +wt2+eps));
        ww2 = sqrt(wt2 ./ (wt1 +wt2+eps));
    else 
        disp('The weight will be equal 1');
        wt1 = ones(size(dxdy1));
        wt2 = ones(size(dxdy2));
        ww1 = ones(size(dxdy1));
        ww2 = ones(size(dxdy2));
    end
end

%%
function [postMap, ss] = EigDecBlock(img, sigma)

    winSize = ceil(sigma*6);
    if ~mod(winSize, 2),
        winSize = winSize + 1;
    end
    
    h = fspecial('gauss', [winSize winSize], sigma);
    
    [hh, ww] = size(img); 
    ss = zeros(2, hh, ww);

    dx = real(img);
    dy = imag(img);
    dxx = imfilter(dx.*dx, h, 'symmetric');
    dxy = imfilter(dx.*dy, h, 'symmetric');
    dyy = imfilter(dy.*dy, h, 'symmetric');
    
    A = ones(size(img));
    B = -(dxx+dyy);
    C = dxx.*dyy - dxy.*dxy;
    
    ss(1, :, :) = abs((-B+sqrt(B.^2-4*A.*C))./(2*A));
    ss(2, :, :) = abs((-B-sqrt(B.^2-4*A.*C))./(2*A));
    
    
    V12 = (dxx-dyy + sqrt((dxx-dyy).^2+4*dxy.*dxy))./(2*dxy+eps);
    
    
    postMap = sqrt(squeeze(ss(1, :, :))).*(V12 + 1i)./sqrt(V12.^2+1+eps);
   
end