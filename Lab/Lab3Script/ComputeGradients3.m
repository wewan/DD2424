function [grad_W,grad_b] = ComputeGradients3(X,h,S,S_hat,Y, P, W, GDparams,M_bn,V_bn)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X (d x N)
% Y (K x N)
% P (K x N)
lambda =GDparams.lambda;
IFbn = GDparams.IFbn;
N = size(X,2);
% m = size(S1,1);
% [K,~] = size(W);
% [d,~] = size(X);
% grad_W1 = zeros(m,d);
% grad_b1 = zeros(m,1);
% grad_W2 = zeros(K,m);
% grad_b2 = zeros(K,1);
% calculate g following the lecture 4
L = size(W,1);
grad_W = cell(L,1);
grad_b = cell(L,1);
for ini = 1:L
    grad_W{ini} = zeros(size(W{ini}));
    grad_b{ini} = zeros(size(W{ini},1),1);
end
if IFbn
     g  = cell(N,1);
     % calculate gk
     for i = 1:N
        Yn = Y(:,i); % (K x 1)
        Pn = P(:,i); % (K x 1)
%         Xn = X(:,i); % (d x 1)
        g{i} = - Yn'/(Yn'*Pn)*(diag(Pn)-Pn*Pn');
%         g{i} = -(Yn-Pn)';
        % gradient L w.r.t b{L} = g
        grad_b{L} = grad_b{L} +g{i}';
        grad_W{L} = grad_W{L} +g{i}'*h{L-1}(:,i)';  
     end
     % get grad_bk grad_wk
        grad_b{L} = grad_b{L}/N;
        grad_W{L} = grad_W{L}/N+2*lambda*W{L};
     % propagate to previous layers
     for i = 1:N
        g{i} = g{i}*W{L};
        g{i} = g{i}*diag(S{L-1}(:,i)>0);
     end
     % bn
     for i = L-1:-1:2
         g = BatchNormBackPass(g,S_hat{i},M_bn{i},V_bn{i});
         for j = 1:N
             grad_b{i} = grad_b{i} + g{j}';
             grad_W{i} = grad_W{i} + g{j}'*h{i-1}(:,j)';
         end
         grad_b{i} = grad_b{i}/N;
         grad_W{i} = grad_W{i}/N+2*lambda*W{i};
         
         for m = 1:N
            g{m} = g{m}*W{i};
            g{m} = g{m}*diag(S{i-1}(:,m)>0);
         end
     end
     g = BatchNormBackPass(g,S_hat{1},M_bn{1},V_bn{1});
     for j = 1:N
        grad_b{1} = grad_b{1} + g{j}';
        grad_W{1} = grad_W{1} + g{j}'*X(:,j)';
     end
     grad_b{1} = grad_b{1}/N;
     grad_W{1} = grad_W{1}/N+2*lambda*W{1};
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
else 
    for i = 1:N
        Yn = Y(:,i); % (K x 1)
        Pn = P(:,i); % (K x 1)
        Xn = X(:,i); % (d x 1)
    %   g = - Yn'/(Yn'*Pn)*(diag(Pn)-Pn*Pn');
        g = -(Yn-Pn)';
        % gradient L w.r.t b{L} = g
        for j = (L):-1:2
            grad_b{j} = grad_b{j} +g';
            grad_W{j} = grad_W{j} +g'*h{j-1}(:,i)';  
            % update g
            % (1 x m)
            g = g*W{j};
            g = g*diag(S{j-1}(:,i)>0);
        end
        % assumingg relu activation
        % (1 x m)--> (1 x m x m x m)
        grad_b{1} = grad_b{1} + g';
        grad_W{1} = grad_W{1} + g'*Xn';
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    % gradient J
    for i = 1:L
        grad_W{i} = (1/N)*grad_W{i}+2*lambda*W{i};
        grad_b{i} = (1/N)*grad_b{i};
    end
end

end


function g = BatchNormBackPass(g, S, mu, var)
    eps  = 1e-6;
    g_vb = 0;
    g_mub= 0;
    V_b  = diag(var + eps);
    n    = size(g,1);
    for i = 1:n      
        g_vb = g_vb - 0.5*g{i}*V_b^(-1.5)*diag(S(:,i) - mu);
        g_mub = g_mub - g{i}*V_b^(-0.5);
    end
    for i = 1:n
        g{i} = g{i}*V_b^(-0.5)+2/n*g_vb*diag(S(:,i)-mu)+ g_mub/n;
    end
  
end