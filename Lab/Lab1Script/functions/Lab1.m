% A = load('data_batch_1.mat');
% m = A.data;
% I = reshape(A.data', 32, 32, 3, 10000);
% I = permute(I, [2, 1, 3, 4]);
% montage(I(:, :, :, 1:500), 'Size', [5,5]);

mu = [1 2];
sigma = [1 0.5; 0.5 2];
R = chol(sigma);
z = repmat(mu,10000,1) + randn(10000,2,'double')*R
mean(z)