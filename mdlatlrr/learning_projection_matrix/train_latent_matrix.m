function L = train_latent_matrix(B, B_smooth, col, col_smo)
% Train the extractor of salient features 'L'
%   min_Z,L,E ||Z||_* + ||L||_* +¡¡lambda||E||_1,
%           s.t. X = XZ + LX +E.

[s1_d, s2_d] = size(B);
[s1_s, s2_s] = size(B_smooth);
B_d = B(:,randperm(s2_d, col-col_smo));
B_s = B_smooth(:,randperm(s2_s, col_smo));
B_rand = cat(2,B_d, B_s);
B_rand = B_rand(:, randperm(col));

lambda = 0.4;
disp('Start-latent lrr');
disp('It takes almost 10 min.');
tic
[Z,L,E] = latent_lrr(B_rand,lambda);
toc
disp('Done-latent lrr');

end




