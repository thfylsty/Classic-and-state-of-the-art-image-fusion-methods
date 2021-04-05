function F = IJF(Sa,Sb)
% if ~exist('sigma_r','var')
%     sigma_r = 0.2;
% end

%% normalization
GA = im2double(Sa);
GB = im2double(Sb);
%% Smoothing
r = 3;    lambda = 0.01;
[hei, wid] = size(GA);
N = boxfilter(ones(hei, wid), r);
Ga = smoothing(GA, r, lambda,N);
Gb = smoothing(GB, r, lambda,N);
% r = 7;
N = boxfilter(ones(hei, wid), r);
%% Structure
h = [1 -1];
% MA = Ga.*0;
% MB = MA;
% MAx = diff(Ga,1,1);
% MAx(hei, wid) = 0;
% MAy = diff(Ga,1,2);
% MAy(hei, wid) = 0;
% MA = abs(MAx) + abs(MAy);
% 
% MBx = diff(Gb,1,1);
% MBx(hei, wid) = 0;
% MBy = diff(Gb,1,2);
% MBy(hei, wid) = 0;
% MB = abs(MBx) + abs(MBy);

MA = abs(conv2(Ga,h,'same')) + ...
     abs(conv2(Ga,h','same'));
MB = abs(conv2(Gb,h,'same')) + ...
     abs(conv2(Gb,h','same'));
D = MA - MB;
IA = boxfilter(D,r) ./ N>0;
% imwrite(double(IA),'test.bmp')
%% IJF by blf
% Ga = GA;
% switch(Filter)
%     case 'BF'
%         minA = min(min(GA)); maxA = max(max(GA));
%         for t = 1:T
%             IA = double(IA > 0.5);
%         %     IA = guidedfilter(GA,IA, sigma_s, sigma_r^2);
%             IA = bilateralFilter(IA,GA,minA,maxA,r,sigma_r);%sigma_r=0.2
% %             imwrite(IA,strcat('ia',num2str(t),'.bmp'));
% %             figure,imshow(IA),colormap jet;colorbar
%         end
%% IJF by gif
%     case 'GF'
%         mean_I = boxfilter(GA, r) ./ N;
%         mean_II = boxfilter(GA.*GA, r) ./ N;
%         var_I = mean_II - mean_I .* mean_I;
%         for t = 1:T
%             IA = double(IA > 0.5);
%             IA = IJF_guided(GA, IA, r, sigma_r^2,N,mean_I,var_I);
%         end
%     case 'DTF'
        for t = 1:3
            IA = double(IA > 0.5);
            IA = RF(IA, 10, 0.2, 1, GA);
        end
%         imwrite(IA,'ia1.bmp');
% end
% 
%% Result
F = IA.*GA + (1-IA).*GB;
F = uint8(255*F);