clc;
clear all;

T = rgb2gray(im2double(imread('Lenna.png')));
X = T(141:140+256, 51:50+256);
X(65:192,65:192) = nan;
img_corrupted = X;

figure;
subplot(2,1,1),imshow(img_corrupted);

fprintf(1, '%d Missing entries\n', nnz(isnan(img_corrupted)));
k = 20;

tic
    [pc,W,data_mean,xr,evals,percentVar]=ppca(img_corrupted,k);
toc

difference = X - xr;
squaredError = difference .^ 2;
meanSquaredError = sum(squaredError(:)) / numel(X);
rmsError = sqrt(meanSquaredError);

fprintf(1, 'err=%f\n',rmsError);
subplot(2,1,2),imshow(xr);