function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X (d x N)
% Y (K x N)
% GDparams (object)
% W (K x d)
% b (K x 1)
% Wstar (K x d)
% bstar (K x 1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initial parameter
N  = size(X,2);
n_batch = GDparams.n_batch;
eta = GDparams.eta;
% n_epochs = GDparams.n_epochs;
Wstar = W; 
bstar = b;
Xtrain = X;
Ytrain = Y;
% run epoch
% for i = 1:n_epochs
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = Xtrain(:, inds);
        Ybatch = Ytrain(:, inds);
        P = EvaluateClassifier(Xbatch,Wstar,bstar);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, Wstar, lambda);
        Wstar = Wstar - eta*grad_W;
        bstar = bstar - eta*grad_b;
    end
% end
