function [P,S,h,S1] = EvaluateClassifier2(X,W,b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% W (K x d)
% x (d x N)
% b (K x 1)
% P (k X N)
% W2 (K x m)
% W1 (m x d)
% b2 (K x 1)
% b1 (m x 1)
W1 = W{1};
W2 = W{2};
b1 = b{1};
b2 = b{2};
b1 = repmat(b1,1,size(X,2));%(m,N)
b2 = repmat(b2,1,size(X,2));%(K,N)
S1 = W1*X + b1; % S1(m,N)
% h = S1.*(S1>0);% ReLu
h = max(0,S1);
S = W2*h +b2;
% softmax
P = exp(S)./repmat(sum(exp(S)),size(W{2},1),1);


 
 