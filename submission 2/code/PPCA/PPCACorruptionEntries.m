clc;
clear all;

img = double(imread('Lenna.png'))/255;
img = img(141:140+256, 51:50+256);
msk = zeros(size(img));
msk(65:192,65:192) = imresize(imread('text.png'), 0.5);
img_corrupted = img;
img_corrupted(msk > 0) = nan;

figure;
subplot(2,1,1),imshow(img_corrupted);

fprintf(1, '%d Corruption entries\n', nnz(isnan(img_corrupted)));
k = 20;

tic
[pc,W,data_mean,xr,evals,percentVar]=ppca(img_corrupted,k);
toc

difference = img - xr;
squaredError = difference .^ 2;
meanSquaredError = sum(squaredError(:)) / numel(img);
rmsError = sqrt(meanSquaredError);

fprintf(1, 'err=%f\n',rmsError);
subplot(2,1,2),imshow(xr);