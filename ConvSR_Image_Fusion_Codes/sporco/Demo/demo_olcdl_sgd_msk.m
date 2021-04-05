% Script demonstrating usage of the olcdl_sgd_msk function.
%
% Author: Brendt Wohlberg <brendt@lanl.gov>
%         Jialin Liu <danny19921123@gmail.com>
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'Copyright' and 'License' files
% distributed with the library.


% NB: This script uses only 5 training images to avoid running for
% longer than is appropriate for a demo script. Note, however, that
% the advantages of the online learning algorithms relative to the
% batch algorithms are not apparent for such a small training set.


% Training images
S0 = zeros(512, 512, 5, 'single');
S0(:,:,1) = single(stdimage('lena.grey')) / 255;
S0(:,:,2) = single(stdimage('barbara.grey')) / 255;
S0(:,:,3) = single(stdimage('kiel.grey')) / 255;
S0(:,:,4) = single(rgb2gray(stdimage('mandrill'))) / 255;
tmp = single(stdimage('man.grey')) / 255;
S0(:,:,5) = tmp(101:612, 101:612);


%Reduce images size to speed up demo script
tmp = zeros(256, 256, 5, 'single');
for k = 1:size(S0,3),
  tmp(:,:,k) = imresize(S0(:,:,k), 0.5);
end
S0 = tmp;


% Apply 10% salt & pepper noise to images
[Sn, wn] = spnoise(S0, 0.1);


% Filter input images and compute highpass images
Sl = zeros(size(Sn), 'single');
Sh = zeros(size(Sn), 'single');
opt_filt = [];
opt_filt.Verbose = 0;
opt_filt.MaxMainIter = 500;
opt_filt.RelStopTol = 1e-3;
for index = 1:size(Sn,3),
    sn = Sn(:,:,index);
    opt_filt.DatFidWeight = ~wn(:,:,index);
    sl = l2tvdenoise(sn, 1, opt_filt);
    sh = sn - sl;
    Sl(:,:,index) = sl;
    Sh(:,:,index) = sh;
end


% Construct initial dictionary
D0 = zeros(8,8,32, 'single');
D0(3:6,3:6,:) = single(randn(4,4,32));


% Set up olcdl parameters
lambda = 0.2;
opt = [];
opt.MaxMainIter = 5*size(S0,3);

% Do dictionary learning
W = ~wn;
[D, optinf] = olcdl_sgd_msk(D0, Sh, lambda, W, opt);


% Display learned dictionary
figure;
imdisp(tiledict(D));

% Plot functional value evolution
figure;
plot(optinf.itstat(:,2));
xlabel('Iterations');
ylabel('Functional value');
