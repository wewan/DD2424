function P = EvaluateClassifier(X,W,b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% W (K x d)
% x (d x N)
% b (K x 1)
% P (k X N)
 b = repmat(b,1,size(X,2));
 S = W*X + b; % S(K,N)
 % softmax
 P = exp(S)./repmat(sum(exp(S)),size(W,1),1);
 
 