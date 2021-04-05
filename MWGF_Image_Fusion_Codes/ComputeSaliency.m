%% -------------
function s = ComputeSaliency(img, sigma, alpha)
    
    % --------- Check the input --------
    if (1 ~= size(img, 3)),
        error('The input image should be GRAY.');
    end
    
    [dx, dy] = GradientMethod(double(img), 'zhou'); 
    grad = dx +1j*dy;
    
    [~, cc] = EigDecBlock(grad, sigma);
    wt = sqrt((sqrt(cc(1,:,:))+sqrt(cc(2,:,:))).^2 + alpha*(sqrt(cc(1,:,:))-sqrt(cc(2,:,:))).^2);
    s = squeeze(wt);
end



%% ------------------------------------------------
% [c11, c12    [dxx, dxy
%  c21, c22] =  dyx, dyy]
% B = -(c11+c22), C = c11*c22-c12*c21
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

