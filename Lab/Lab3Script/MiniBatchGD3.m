function [Wstar, bstar,M_av,V_av] = MiniBatchGD3(X, Y, GDparams, W, b,varargin)
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
L = size(W,1);
n_batch = GDparams.n_batch;
Wstar = W;
bstar = b;
Xtrain = X;
Ytrain = Y;
V_w = cell(L,1);
V_b = cell(L,1);

if numel(varargin) ==2 
    M_av = varargin{1};
    V_av = varargin{2};
else
    M_av = cell(L-1,1);
    V_av = cell(L-1,1);
end

if GDparams.ifmomentum
    for i = 1:L
        V_w{i} = zeros(size(Wstar{i}));
        V_b{i} = zeros(size(bstar{i}));
    end
end
% run epoch
% for i = 1:n_epochs
    for j=1:fix(N/n_batch)
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = Xtrain(:, inds);
        Ybatch = Ytrain(:, inds);
        [P,S,S_hat,h,M_bn,V_bn] = EvaluateClassifier3(Xbatch,Wstar,bstar,GDparams);
        if GDparams.IFbn
            if numel(varargin) == 0 && j == 1
                M_av = M_bn;
                V_av = V_bn;
            else
                for i = 1:L-1
                    M_av{i} = GDparams.alpha*M_av{i} +(1-GDparams.alpha)*M_bn{i};
                    V_av{i} = GDparams.alpha*V_av{i} +(1-GDparams.alpha)*V_bn{i};
                end
            end

            [grad_W,grad_b] = ComputeGradients3(Xbatch,h,S,S_hat,Ybatch, P, Wstar, GDparams,M_bn,V_bn);
        else
            [grad_W,grad_b] = ComputeGradients3(Xbatch,h,S,S_hat,Ybatch, P, Wstar, GDparams);
        end
        
        
        if GDparams.ifmomentum
            for i = 1:L
                V_w{i} = GDparams.rho*V_w{i} + GDparams.eta*grad_W{i};
                V_b{i} = GDparams.rho*V_b{i} + GDparams.eta*grad_b{i};
                Wstar{i} = Wstar{i}-V_w{i};
                bstar{i} = bstar{i}-V_b{i};
            end
        else
            for i = 1:L
                Wstar{i} = Wstar{i}-GDparams.eta*grad_W{i};
                bstar{i} = bstar{i}-GDparams.eta*grad_b{i};   
            end  
        end

    end

% end
