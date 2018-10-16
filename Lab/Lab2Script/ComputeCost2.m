function  J = ComputeCost2(X,Y,W,b, lambda)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X (d x N)
% y (N x 1) or (K x N)
% P (K x N)
% to judge whether it is vector or onehot
if  size(Y,1) == size(X,2) && size(Y,2)== 1
    % change Y (N x 1) to Y (K x N)
    Y = zeros(length(unique(Y)),length(Y));
    for i = 1: length(y)
        Y(y(i),i) = 1;
    end
end
% Y (K x N)
[P,~,~,~] = EvaluateClassifier2(X,W,b);
D = size(X,2);
% Y'P ---> (1 x K) x (K x 1) if Y is a vector
% Y.*p for extracting the matched P if Y is a matrix. 
J  = -1/D*sum(log(sum(Y.*P)))+lambda*(sum(sum(W{1}.^2))+sum(sum(W{2}.^2)));
% J  = sum(diag(-log(Y'*P)))/size(X,2) + lambda*(sum(sum(W{1}.^2))+sum(sum(W{2}.^2)));
% if J1 ==J
%     sprintf("equal!")
% else
%     sprintf("not equal!")
% end