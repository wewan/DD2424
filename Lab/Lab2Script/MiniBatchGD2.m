function [Wstar, bstar] = MiniBatchGD2(X, Y, GDparams, W, b, lambda)
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

% n_epochs = GDparams.n_epochs;
Wstar1 = W{1}; 
bstar1 = b{1};
Wstar2 = W{2}; 
bstar2 = b{2};
Wstar = {Wstar1,Wstar2};
bstar = {bstar1,bstar2};
Xtrain = X;
Ytrain = Y;
if GDparams.ifmomentum
    V_w1 = zeros(size(Wstar1));
    V_w2 = zeros(size(Wstar2));
    V_b1 = zeros(size(bstar1));
    V_b2 = zeros(size(bstar2));
end
% run epoch
% for i = 1:n_epochs
    for j=1:fix(N/n_batch)
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = Xtrain(:, inds);
        Ybatch = Ytrain(:, inds);
%         P = EvaluateClassifier2(Xbatch,Wstar,bstar);
        [P,~,h,S1] = EvaluateClassifier2(Xbatch,Wstar,bstar);
%         [grad_W, grad_b] = ComputeGradients2(Xbatch, Ybatch, P, Wstar, lambda);
        [grad_W,grad_b] = ComputeGradients2(Xbatch,h,S1,Ybatch,P, Wstar, lambda);
        
        if GDparams.ifmomentum
            V_w1 = GDparams.rho*V_w1 + GDparams.eta*grad_W{1};
            V_w2 = GDparams.rho*V_w2 + GDparams.eta*grad_W{2};
            V_b1 = GDparams.rho*V_b1 + GDparams.eta*grad_b{1};
            V_b2 = GDparams.rho*V_b2 + GDparams.eta*grad_b{2};
            
            Wstar1 = Wstar1 - V_w1;
            Wstar2 = Wstar2 - V_w2;
            bstar1 = bstar1 - V_b1;
            bstar2 = bstar2 - V_b2;
            
            Wstar = {Wstar1,Wstar2};
            bstar = {bstar1,bstar2};
        else
            Wstar1 = Wstar1 - GDparams.eta*grad_W{1};
            bstar1 = bstar1 - GDparams.eta*grad_b{1};
            Wstar2 = Wstar2 - GDparams.eta*grad_W{2};
            bstar2 = bstar2 - GDparams.eta*grad_b{2};
            Wstar = {Wstar1,Wstar2};
            bstar = {bstar1,bstar2};
        end

    end

% end
