%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all; clear;

%% Exercise 1 Upgrade Assignment 2 code to train & test k-layer networks
[train_X, train_Y, train_y] = LoadBatch('data_batch_1.mat');
[valid_X, valid_Y, valid_y] = LoadBatch('data_batch_2.mat');
[test_X,  test_Y,  test_y ] = LoadBatch('test_batch.mat');
% mean of dimensions of X
mean_data = mean(train_X,2);
train_X = train_X -repmat(mean_data,[1,size(train_X,2)]);
valid_X = valid_X -repmat(mean_data,[1,size(train_X,2)]);
test_X  = test_X  -repmat(mean_data,[1,size(train_X,2)]);
% initialize W & b 
rng_number   = 40;      % seed of random generator
layers       = 2;       % number of total layers
hidden_notes = [50];    % the nodes of hidden layers
n_epochs     = 10;       % number of epochs
n_batch      = 100;     % the batch size 
eta          = 0.003;     % the learning rate 
check        = false;   % show the mean and variance of w b
lambda       = 1.2617e-4;   % the weight of regularization
ifplot       = true;   % plot the layers 
rho          = 0.9;    % the rate of momentum 
ifmomentum   = true;   % whether add momentum
decay_rate   = 0.95;     % decay rate
IFbn         = false;
alpha        = 0.99;
[W,b,GDparams]= ...
    ParameterInit3(train_X ,...
                  train_Y,...
                  layers , ...
                  hidden_notes ,...
                  rng_number,...
                  n_epochs,...
                  n_batch,...
                  eta,...
                  rho,...
                  check,...
                  ifplot,...
                  ifmomentum,...
                  decay_rate,...
                  IFbn,...
                  lambda,...
                  alpha);
% evaluate classifier

%% Test the gradient accuracy
%
tic
lambda = 0;
Ttrain_X = train_X(:,1:20); 
Ttrain_Y = train_Y(:,1:20);
[P,S,S_hat,h,M_bn,V_bn] = EvaluateClassifier3(Ttrain_X,W,b,GDparams);
J = ComputeCost3(Ttrain_X,Ttrain_Y,W,b, GDparams);
[grad_W,grad_b] = ComputeGradients3(Ttrain_X,h,S,S_hat,Ttrain_Y, P, W, GDparams,M_bn,V_bn);
if true
    [ngrad_b, ngrad_W] = ComputeGradsNumSlow(Ttrain_X, Ttrain_Y, W, b, GDparams, 1e-6);
    save('ngrad_b.mat','ngrad_b');
    save('ngrad_W.mat','ngrad_W');
else
    ngrad_b = load('ngrad_b');
    ngrad_b = ngrad_b.ngrad_b;
    ngrad_W = load('ngrad_W'); 
    ngrad_W = ngrad_W.ngrad_W;
end
%test
diff_W1 = norm(grad_W{1}-ngrad_W{1})/max([1e-6,norm(grad_W{1})+norm(ngrad_W{1})]);
sprintf('the difference of gradient W1 between two method is %f',diff_W1)
diff_b1 = norm(grad_b{1}-ngrad_b{1})/max([1e-6,norm(grad_b{1})+norm(ngrad_b{1})]);
sprintf('the difference of gradient b1 between two method is %f',diff_b1)
diff_W2 = norm(grad_W{2}-ngrad_W{2})/max([1e-6,norm(grad_W{2})+norm(ngrad_W{2})]);
sprintf('the difference of gradient W2 between two method is %f',diff_W2)
diff_b2 = norm(grad_b{2}-ngrad_b{2})/max([1e-6,norm(grad_b{2})+norm(ngrad_b{2})]);
sprintf('the difference of gradient b2 between two method is %f',diff_b2)
diff_W3 = norm(grad_W{3}-ngrad_W{3})/max([1e-6,norm(grad_W{3})+norm(ngrad_W{3})]);
sprintf('the difference of gradient W3 between two method is %f',diff_W3)
diff_b3 = norm(grad_b{3}-ngrad_b{3})/max([1e-6,norm(grad_b{3})+norm(ngrad_b{3})]);
sprintf('the difference of gradient b3 between two method is %f',diff_b3)
toc
%}

% run mini batch GD to test gradients computation
%
trainD = {train_X(:,1:100), train_Y(:,1:100), train_y(1:100)};
validD = {valid_X(:,1:100), valid_Y(:,1:100), valid_y(1:100)};
testD  = {test_X(:,1:100),  test_Y(:,1:100),  test_y(1:100)}; 
tic
% trainRecord{1} is cost and trainRecord{2}is accuracy
[trainRecord,validRecord]= Run3(trainD,validD,GDparams,W,b);
toc
%}

%% Exercise 2
%
trainD = {train_X, train_Y, train_y};
validD = {valid_X, valid_Y, valid_y};
testD  = {test_X,  test_Y,  test_y}; 

tic

Run3(trainD,validD,GDparams,W,b);

toc
%}

%% Exercise 3 Training network
%
% coarse-to-fine

% GDparams.ifmomentum   = true;

trainD = {train_X, train_Y, train_y};
validD = {valid_X, valid_Y, valid_y};
testD  = {test_X,  test_Y,  test_y}; 
% set range
TRY_TIMES = 20;
eta_collection = zeros(TRY_TIMES,1);
lambda_collection = zeros(TRY_TIMES,1);
TrainAcc_collection = zeros(TRY_TIMES,1);
TestAcc_collection = zeros(TRY_TIMES,1);
Train_curve = cell(TRY_TIMES,1);
Test_curve = cell(TRY_TIMES,1);
% Test

e_max = 0.04;
e_min = 0.008;
lambda_max = -3;
lambda_min = -7;

for i = 1:TRY_TIMES
%     GDparams.eta =10^(e_min +(e_max-e_min)*rand(1,1));
    GDparams.eta = e_min +(e_max-e_min)*rand(1,1);
    lambda = lambda_min +(lambda_max-lambda_min)*rand(1,1);
    lambda = 10^lambda;
    eta_collection(i) = GDparams.eta;
    lambda_collection(i) = lambda;
    fprintf('-------------------%d/%d---------------------\n',i,TRY_TIMES);
    [trainRecord,validRecord]= Run3(trainD,validD,GDparams,W,b);
    Train_curve{i} = trainRecord;
    Test_curve{i}  = validRecord;
    TrainAcc_collection(i) = trainRecord{2}(end);
    TestAcc_collection(i) = validRecord{2}(end);  
end
% sort
[TestAcc_collection,I] = sort(TestAcc_collection);
TrainAcc_collection = TrainAcc_collection(I);
eta_collection = eta_collection(I);
lambda_collection = lambda_collection(I);
Train_curve = {Train_curve{I}};
Test_curve  = {Test_curve{I}};
%save
Pairs_Data = {
'eta_collection' ,eta_collection;...
'lambda_collection' ,lambda_collection ;...
'TrainAcc_collection' ,TrainAcc_collection ;...
'TestAcc_collection' ,TestAcc_collection ;...
'Train_curve',Train_curve;...
'Test_curve',Test_curve};
save('Pairs_Data','Pairs_Data');
%}
%
% plot first 3
Pairs_Data = load('Pairs_Data.mat');
for i = 1:3
    
    train_cost = Pairs_Data.Pairs_Data{5,2}{end+1-i}{1};
    valid_cost = Pairs_Data.Pairs_Data{6,2}{end+1-1}{1};
    lambda = Pairs_Data.Pairs_Data{2,2}(end+1-i);
    Train_ACC = Pairs_Data.Pairs_Data{3,2}((end+1-i));
    Test_ACC  = Pairs_Data.Pairs_Data{4,2}((end+1-i));
    eta = Pairs_Data.Pairs_Data{1,2}(end+1-i);
    figure
    plot(0:1:GDparams.n_epochs,train_cost)
    hold on 
    plot(0:1:GDparams.n_epochs,valid_cost)
    legend('train cost','valid cost')
    xlabel('epoch')
    ylabel('cost')
    title(  ['lambda=',num2str(lambda),...
              ' eta=',num2str(eta),...
              ' TrainACC=',num2str(Train_ACC),...
              ' TestACC=',num2str(Test_ACC)]);
end

%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% function
function acc = ComputeAccuracy3(X,y,W,b,GDparams,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X (d x N)
% y (N x 1)
% W (K x d)
% b (K x 1)

% method 1
[P,~,~,~,~,~] = EvaluateClassifier3(X,W,b,GDparams,varargin);
[~,Index] = max(P);
acc = sum(Index == y')/length(y);
% method 2
% use onehot Y .* onehot outcome and then sum up
end

function  J = ComputeCost3(X,Y,W,b, GDparams,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda = GDparams.lambda;
[P,~,~,~,~,~] = EvaluateClassifier3(X,W,b,GDparams,varargin);
D = size(X,2);
% Y'P ---> (1 x K) x (K x 1) if Y is a vector
% Y.*p for extracting the matched P if Y is a matrix. 
J1  = -1/D*sum(log(sum(Y.*P)));
J2  = 0;
L   = size(W,1);
for i = 1:L
    J2 = J2 + lambda*(sum(sum(W{i}.^2)));
end
J = J1 + J2;
end
        
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
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


function[X,Y,y] = LoadBatch(filename)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Imagedata = load(filename);
% X (d x N) d:dimensionality of each image N:image number
X = double(Imagedata.data')./255;
% y (N x 1) is the vector label of images 
y = double(Imagedata.labels+1);
% Y (k x N) k is class number
Y = double(zeros(length(unique(y)),length(y)));
for i = 1: length(y)
    Y(y(i),i) = 1;
end
end

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

end

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

function [trainRecord,validRecord]= Run3(trainP,validP,GDparams,W,b)
%sparse parameter 
train_X = trainP{1};
train_Y = trainP{2};
train_y = trainP{3};
valid_X = validP{1};
valid_Y = validP{2};
valid_y = validP{3};


% set cost and acc
train_cost = zeros(GDparams.n_epochs+1,1);
train_acc  = zeros(GDparams.n_epochs+1,1);
valid_cost = zeros(GDparams.n_epochs+1,1);
valid_acc  = zeros(GDparams.n_epochs+1,1);
Wstar = W;
bstar = b;

% add initial number
eta = GDparams.eta ;
train_cost(1) = ComputeCost3(train_X,train_Y,Wstar,bstar, GDparams);
train_acc(1)  = ComputeAccuracy3(train_X,train_y,Wstar,bstar,GDparams);
valid_cost(1) = ComputeCost3(valid_X,valid_Y,Wstar,bstar, GDparams);
valid_acc(1)  = ComputeAccuracy3(valid_X,valid_y,Wstar,bstar,GDparams);

% run epoch
for i = 1:GDparams.n_epochs
    if rem(i,1)==0
         GDparams.eta = GDparams.eta*GDparams.decay_rate;
    end
    if GDparams.IFbn
        if i ==1
            [Wstar, bstar,M_av,V_av] = MiniBatchGD3(train_X, train_Y, GDparams, Wstar, bstar);
        else
            [Wstar, bstar,M_av,V_av] = MiniBatchGD3(train_X, train_Y, GDparams, Wstar, bstar,M_av,V_av);
        end
        train_cost(i+1) = ComputeCost3(train_X,train_Y,Wstar,bstar, GDparams,M_av,V_av);
        train_acc(i+1)  = ComputeAccuracy3(train_X,train_y,Wstar,bstar,GDparams,M_av,V_av);
        valid_cost(i+1) = ComputeCost3(valid_X,valid_Y,Wstar,bstar, GDparams,M_av,V_av);
        valid_acc(i+1)  = ComputeAccuracy3(valid_X,valid_y,Wstar,bstar,GDparams,M_av,V_av);
    else
        [Wstar, bstar,M_av,V_av] = MiniBatchGD3(train_X, train_Y, GDparams, Wstar, bstar);
        train_cost(i+1) = ComputeCost3(train_X,train_Y,Wstar,bstar, GDparams);
        train_acc(i+1)  = ComputeAccuracy3(train_X,train_y,Wstar,bstar,GDparams);
        valid_cost(i+1) = ComputeCost3(valid_X,valid_Y,Wstar,bstar, GDparams);
        valid_acc(i+1)  = ComputeAccuracy3(valid_X,valid_y,Wstar,bstar,GDparams);
    end
    if train_cost(i+1)>3*train_cost(1)
        fprintf('bad parameter, too large\n');
        break
    end
    fprintf('################### epoch=%3d ###################### \n',i);
    fprintf('train loss=%5f Validation loss = %5f\n',train_cost(i+1), valid_cost(i+1));
    fprintf('train acc=%5f  Validation acc = %5f\n',train_acc(i+1),valid_acc(i+1));
end

trainRecord = {train_cost,train_acc};
validRecord = {valid_cost,valid_acc};

if GDparams.ifplot
    % plot cost
    figure
    plot(0:1:GDparams.n_epochs,train_cost)
    hold on 
    plot(0:1:GDparams.n_epochs,valid_cost)
    legend('train cost','valid cost')
    xlabel('epoch')
    ylabel('cost')
    title(  ['lambda=',  num2str(GDparams.lambda),...
            ' epochs=',num2str(GDparams.n_epochs),...
            ' batch=', num2str(GDparams.n_batch),...
            ' eta=',     num2str(eta),...
            ' rho=',     num2str(GDparams.rho)]);
    % plot W
   %{
        for i=1:10
        im = reshape(Wstar{j}(i, :), 32, 32, 3);
        s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
        s_im{i} = permute(s_im{i}, [2, 1, 3]);
        end
        figure
        title(  ['lambda=',  num2str(lambda),...
                'epochs=',num2str(GDparams.n_epochs),...
                'batch=', num2str(GDparams.n_batch),...
                'eta=',     num2str(GDparams.eta)]);
        montage(s_im, 'Size', [2,5]);
  %}
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% function append
function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

[c, ~] = ComputeCost(X, Y, W, b, lambda);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        [c2, ~] = ComputeCost(X, Y, W, b_try, lambda);
        grad_b{j}(i) = (c2-c) / h;
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})   
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        [c2, ~] = ComputeCost(X, Y, W_try, b, lambda);
        
        grad_W{j}(i) = (c2-c) / h;
    end
end
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);
for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost3(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost3(X, Y, W, b_try, lambda);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost3(X, Y, W_try, b, lambda);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost3(X, Y, W_try, b, lambda);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
end
