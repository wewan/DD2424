function grads = ComputeGradients(X,Y,RNN,a,h,p)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% h --->(m,n+1) 
f = fieldnames(RNN)';
for i=1:numel(f)
  grads.(f{i}) = zeros(size(RNN.(f{i})));
end
n = size(X,2);
for i =n:-1:1
    yn = Y(:,i);
    pn = p(:,i);
    % g -->(1,K)
    g = -(yn - pn)';
    grad_ot = g;
    % c -->(K,n)
    grads.c = grads.c +g'; 
    % 
    grads.V = grads.V + g'*h(:,i+1)';
    if i == n
        grad_h = grad_ot*RNN.V;
        grad_a = grad_h*diag(1-tanh(a(:,i)).^2);
    else
        grad_h = grad_ot*RNN.V+grad_a*RNN.W;
        grad_a = grad_h*diag(1-tanh(a(:,i)).^2);
    end
    grads.b = grads.b + grad_a';
    xn = X(:,i);
    grads.W = grads.W +grad_a'*h(:,i)';
    grads.U = grads.U +grad_a'*xn';
end

end