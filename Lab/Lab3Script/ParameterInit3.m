function [W,b,GDparams]= ParameterInit3(X,Y,layers,hidden_notes,rng_number,n_epochs,n_batch,eta,rho,check,ifplot,ifmomentum,decay_rate,IFbn,lambda,alpha)
% hidden_notes = m
% X  (d x N)
% y  (N x 1)
% Y  (K x N)
% W2 (K x m)
% W1 (m x d)
% b2 (K x 1)
% b1 (m x 1)
% P  (K x N)
rng(rng_number);
mu = 0;
sigma = 0.001;
% initialize w and b
W = cell(layers,1);
b = cell(layers,1);
L = layers;
% M is the numbers of nodes of each layer
M = [size(X,1),hidden_notes,size(Y,1)];%(L+1,1)
for i = 1:L
    W{i} = mu + sigma*randn(M(i+1),M(i),'double');%W1 (m x d)
    b{i} = double(zeros(M(i+1),1));
end
% initialize GDparams 
GDparams.n_batch    = n_batch;
GDparams.eta        = eta;
GDparams.n_epochs   = n_epochs;
GDparams.rho        = rho;
GDparams.ifplot     = ifplot;
GDparams.ifmomentum = ifmomentum;
GDparams.decay_rate = decay_rate;
GDparams.IFbn       = IFbn;
GDparams.lambda     = lambda;
GDparams.alpha      = alpha;
% check answer
if check
    sprintf('mean of W1 is %f',mean2(W{1}))
    sprintf('standard deviation of W1 is %f',std2(W{1}))
    sprintf('mean of W2 is %f',mean2(W{2}))
    sprintf('standard deviation of W2 is %f',std2(W{2}))
end
end