%% training for deep
clear all
clc

disp('Training your own projection matrices.');

training_path = './training_data/';
col = 2000;
col_smooth = 1000;

unit = 8;
[B_detail1, B_smooth1] = train_choose_detail(training_path, unit);
% display these image patches
% display_image_patches(B_detail1, B_smooth1, unit);
L_8 = train_latent_matrix(B_detail1, B_smooth1, col, col_smooth);
save_path = ['../models/L_8_detail_mix_',num2str(col),'_s',num2str(col_smooth),'.mat'];
save(save_path, 'L_8');

unit = 16;
[B_detail2, B_smooth2] = train_choose_detail(training_path, unit);
% display these image patches
% display_image_patches(B_detail2, B_smooth2, unit);
L_16 = train_latent_matrix(B_detail2, B_smooth2, col, col_smooth);
save_path = ['../models/L_16_detail_mix_',num2str(col),'_s',num2str(col_smooth),'.mat'];
save(save_path, 'L_16');










