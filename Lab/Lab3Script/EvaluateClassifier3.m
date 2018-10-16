function [P,S,S_hat,h,M_bn,V_bn] = EvaluateClassifier3(X,W,b,GDparams,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L is number of layers
% W -- {} Lx1 cell
% b -- {} Lx1 cell
% S -- {} Lx1 cell
% h -- {} (L-1)x1 cell 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IFbn = GDparams.IFbn;
L = size(W,1);
S = cell(L,1);
S_hat = cell(L,1);
h = cell(L-1,1);  
n = size(X,2);
S{1} = W{1}*X +repmat(b{1},1,n);
S_hat{1} = S{1};
if numel(varargin) == 2
    M_bn = varargin{1};
    V_bn = varargin{2};
else
    M_bn = cell(L-1,1);
    V_bn = cell(L-1,1);
end

for i = 1:L-1
    if IFbn
        if numel(varargin) == 2
            mean_score = varargin{1}{i};
            var_score  = varargin{2}{i};
            S_hat{i} = S{i};
            [S{i},M_bn{i},V_bn{i}]= BatchNormalization(S{i},mean_score,var_score);
        else
            S_hat{i} = S{i};
            [S{i},M_bn{i},V_bn{i}]= BatchNormalization(S{i});
        end
    end
    h{i} = max(0,S{i});
    S{i+1}= W{i+1}*h{i}+repmat(b{i+1},1,n);  
end
S_hat{L} = S{L};
P = exp(S{L})./repmat(sum(exp(S{L})),size(W{L},1),1);
    
end

function [s_bn,mean_scores,var_scores] = BatchNormalization(scores,varargin)
    if numel(varargin) == 2
        eps = 1e-6;
        mean_scores = varargin{1};
        var_scores  = varargin{2};
        s_bn = diag(var_scores+eps)^(-0.5) *(scores-repmat(mean_scores,1,size(scores,2)));    
    else
        eps = 1e-6;
        n = size(scores,2);
        mean_scores = mean(scores,2);
        var_scores = var(scores, 0, 2);
        var_scores = var_scores *(n-1)/n;
        s_bn = diag(var_scores+eps)^(-0.5) *(scores-repmat(mean_scores,1,size(scores,2))); 
    end
    
end

