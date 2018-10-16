%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X  (d x N)
% y  (N x 1)
% Y  (K x N)
% W2 (K x m)
% W1 (m x d)
% b2 (K x 1)
% b1 (m x 1)
% P  (K x N)
clc; close all; clear;

%% Exercise 1 zero mean and initial W&b
[train_X, train_Y, train_y] = LoadBatch('data_batch_1.mat');
[valid_X, valid_Y, valid_y] = LoadBatch('data_batch_2.mat');
[test_X,  test_Y,  test_y ] = LoadBatch('test_batch.mat');
% mean of dimensions of X
mean_data = mean(train_X,2);
train_X = train_X -repmat(mean_data,[1,size(train_X,2)]);
valid_X = valid_X -repmat(mean_data,[1,size(train_X,2)]);
test_X  = test_X  -repmat(mean_data,[1,size(train_X,2)]);
% initialize W & b 
rng_number   = 40;
hidden_notes = 50;
n_epochs     = 10;
n_batch      = 100;
eta          = 0.1;
check        = false;
lambda       = 0;
ifplot       = false;
rho          = 0.89;
ifmomentum   = false;
decay_rate   = 0.94;
[W,b,GDparams]= ...
    ParameterInit(train_X,...
                  train_Y,...
                  hidden_notes,...
                  rng_number,...
                  n_epochs,...
                  n_batch,...
                  eta,...
                  rho,...
                  check,...
                  ifplot,...
                  ifmomentum,...
                  decay_rate);

%% Exercise 2 compute the gradients for the network parameters
%
tic
lambda = 0;
Ttrain_X = train_X(:,1:10); 
Ttrain_Y = train_Y(:,1:10);
[P,S,h,S1] = EvaluateClassifier2(Ttrain_X,W,b);
J = ComputeCost2(Ttrain_X,Ttrain_Y,W,b, lambda);
[grad_W,grad_b] = ComputeGradients2(Ttrain_X,h,S1, Ttrain_Y, P, W, lambda);
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(Ttrain_X, Ttrain_Y, W, b, lambda, 1e-5);
%test
diff_W1 = norm(grad_W{1}-ngrad_W{1})/max([1e-6,norm(grad_W{1})+norm(ngrad_W{1})]);
sprintf('the difference of gradient W between two method is %f',diff_W1)
diff_b1 = norm(grad_b{1}-ngrad_b{1})/max([1e-6,norm(grad_b{1})+norm(ngrad_b{1})]);
sprintf('the difference of gradient b between two method is %f',diff_b1)
diff_W2 = norm(grad_W{2}-ngrad_W{2})/max([1e-6,norm(grad_W{2})+norm(ngrad_W{2})]);
sprintf('the difference of gradient W between two method is %f',diff_W2)
diff_b2 = norm(grad_b{2}-ngrad_b{2})/max([1e-6,norm(grad_b{2})+norm(ngrad_b{2})]);
sprintf('the difference of gradient b between two method is %f',diff_b2)
toc
%}

% run mini batch GD to test gradients computation
%
trainD = {train_X(:,1:100), train_Y(:,1:100), train_y(1:100)};
validD = {valid_X(:,1:100), valid_Y(:,1:100), valid_y(1:100)};
testD  = {test_X(:,1:100),  test_Y(:,1:100),  test_y(1:100)}; 
tic
% trainRecord{1} is cost and trainRecord{2}is accuracy
[trainRecord,validRecord]= Run2(trainD,validD,GDparams,W,b,lambda);
toc
%}

%% Exercise 3 Adding momentum
%
trainD = {train_X, train_Y, train_y};
validD = {valid_X, valid_Y, valid_y};
testD  = {test_X,  test_Y,  test_y}; 

GDparams.ifmomentum   = true;
GDparams.rho = 0.5;
tic
%trainRecord{1} is cost and trainRecord{2}is accuracy
Run2(trainD,validD,GDparams,W,b,lambda);

GDparams.rho = 0.7;
Run2(trainD,validD,GDparams,W,b,lambda);

GDparams.rho = 0.8;
Run2(trainD,validD,GDparams,W,b,lambda);

GDparams.rho = 0.9;
Run2(trainD,validD,GDparams,W,b,lambda);

GDparams.rho = 0.99;
Run2(trainD,validD,GDparams,W,b,lambda);
toc
%}

%% Training network
% coarse-to-fine
%
GDparams.ifmomentum   = true;
trainD = {train_X, train_Y, train_y};
validD = {valid_X, valid_Y, valid_y};
testD  = {test_X,  test_Y,  test_y}; 
% set range
TRY_TIMES = 100;
eta_collection = zeros(TRY_TIMES,1);
lambda_collection = zeros(TRY_TIMES,1);
TrainAcc_collection = zeros(TRY_TIMES,1);
TestAcc_collection = zeros(TRY_TIMES,1);
Train_curve = cell(TRY_TIMES,1);
Test_curve = cell(TRY_TIMES,1);
% Test

e_max = 0.04;
e_min = 0.02;
lambda_max = -1;
lambda_min = -4;
tic
for i = 1:TRY_TIMES
    GDparams.eta = (e_min +(e_max-e_min)*rand(1,1));
    lambda = lambda_min +(lambda_max-lambda_min)*rand(1,1);
    lambda = 10^lambda;
    eta_collection(i) = GDparams.eta;
    lambda_collection(i) = lambda;
    fprintf('-------------------%d/%d---------------------\n',i,TRY_TIMES);
    [trainRecord,validRecord]= Run2(trainD,validD,GDparams,W,b,lambda);
    Train_curve{i} = trainRecord;
    Test_curve{i}  = validRecord;
    TrainAcc_collection(i) = trainRecord{2}(end);
    TestAcc_collection(i) = validRecord{2}(end);  
end
toc
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

%% plot first 3  and  trianing for 30 epochs
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
trainD = {train_X, train_Y, train_y};
validD = {valid_X, valid_Y, valid_y};
testD  = {test_X,  test_Y,  test_y}; 
% [train_X1, train_Y1, train_y1] = LoadBatch('data_batch_1.mat');
% [train_X2, train_Y2, train_y2] = LoadBatch('data_batch_3.mat');
% [train_X3, train_Y3, train_y3] = LoadBatch('data_batch_4.mat');
% [train_X4, train_Y4, train_y4] = LoadBatch('data_batch_5.mat');
% [test_X,  test_Y,  test_y ] = LoadBatch('test_batch.mat');
% trainD = {[train_X1,train_X2,train_X3,train_X4], ...
%           [train_Y1,train_Y2,train_Y3,train_Y4],...
%           [train_y1;train_y2;train_y3;train_y4]};
% testD  = {test_X,  test_Y,  test_y}; 
GDparams.n_epochs= 30;
GDparams.ifmomentum   = true;
GDparams.ifplot=true;
GDparams.eta = Pairs_Data.Pairs_Data{1,2}(end);
lambda = Pairs_Data.Pairs_Data{2,2}(end);
[trainRecord,validRecord]= Run2(trainD,testD,GDparams,W,b,lambda);


%%%%%%%%%%%%%%%%%%%%%%%%%  functions  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
end

function acc = ComputeAccuracy2(X,y,W,b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X (d x N)
% y (N x 1)
% W (K x d)
% b (K x 1)

% method 1
[P,~,~,~] = EvaluateClassifier2(X,W,b);
[~,Index] = max(P);
acc = sum(Index == y')/length(y);
% method 2
% use onehot Y .* onehot outcome and then sum up
end

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
% J  = -1/D*sum(log(sum(Y.*P)))+lambda*(sum(sum(W{1}.^2))+sum(sum(W{2}.^2)));
J  = sum(diag(-log(Y'*P)))/size(X,2) + lambda*(sum(sum(W{1}.^2))+sum(sum(W{2}.^2)));
% if J1 ==J
%     sprintf("equal!")
% else
%     sprintf("not equal!")
% end
end

function [grad_W,grad_b] = ComputeGradients2(X,h,S1, Y, P, W, lambda)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X (d x N)
% Y (K x N)
% P (K x N)
% S1 (m x N)
% h  (m x N)
% W2 (K x m)
% W1 (m x d)
% b2 (K x 1)
% b1 (m x 1)
% drad_w1 (m x d)
% grad_b1 (m x 1)
% drad_w2 (K x m)
% grad_b2 (K x 1)
N = size(X,2);
m = size(S1,1);
[K,~] = size(W);
[d,~] = size(X);
grad_W1 = zeros(m,d);
grad_b1 = zeros(m,1);
grad_W2 = zeros(K,m);
grad_b2 = zeros(K,1);
% calculate g following the lecture 4
for i = 1:N
    Yn = Y(:,i); % (K x 1)
    Pn = P(:,i); % (K x 1)
    hn = h(:,i); % (m x 1)
    Xn = X(:,i); % (d x 1)
    S1n= S1(:,i);% (m x 1)
    % according to the Lecture 4 calculate g 
    %( 1 x K )
    g = - Yn'/(Yn'*Pn)*(diag(Pn)-Pn*Pn');
    % gradient L w.r.t b2 = g
    %( K x 1 )
    grad_b2 = grad_b2 +g';
    % gradient L w.r.t W2 = g'h; hn (m x 1)
    %( K x m )-> (K x 1 x 1 x m)
    grad_W2 = grad_W2 +g'*hn';  
    % update g
    % (1 x m)
    g = g*W{2};
    % assumingg relu activation
    % (1 x m)--> (1 x m x m x m)
    g = g*diag(S1n>0);
    grad_b1 = grad_b1 + g';
    grad_W1 = grad_W1 +g'*Xn';
    
end
% gradient J
grad_W1 = (1/N)*grad_W1+ 2*lambda*W{1};
grad_W2 = (1/N)*grad_W2+ 2*lambda*W{2};
grad_b1 = (1/N)*grad_b1;
grad_b2 = (1/N)*grad_b2;
grad_W = {grad_W1,grad_W2};
grad_b = {grad_b1,grad_b2};

end

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
end

function GDparams = GDparamSetting(n_epochs,n_batch,eta)
GDparams.n_batch = n_batch;
GDparams.eta = eta;
GDparams.n_epochs = n_epochs;
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
end

function [W,b,GDparams]= ParameterInit(X,Y,hidden_notes,rng_number,n_epochs,n_batch,eta,rho,check,ifplot,ifmomentum,decay_rate)
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
W1 = mu + sigma*randn(hidden_notes,size(X,1),'double');%W1 (m x d)
W2 = mu + sigma*randn(size(Y,1),hidden_notes,'double');%W2 (K x m)
% b1 = mu + sigma*randn(size(hidden_notes,1),1,'double');
% b2 = mu + sigma*randn(size(Y,1),1,'double');
b1 = double(zeros(hidden_notes,1));
b2 = double(zeros(size(Y,1),1));
W = {W1,W2};
b = {b1,b2};

GDparams.n_batch = n_batch;
GDparams.eta = eta;
GDparams.n_epochs = n_epochs;
GDparams.rho = rho;
GDparams.ifplot = ifplot;
GDparams.ifmomentum = ifmomentum;
GDparams.decay_rate = decay_rate;

% check answer
if check
    sprintf('mean of W1 is %f',mean2(W{1}))
    sprintf('standard deviation of W1 is %f',std2(W{1}))
    sprintf('mean of W2 is %f',mean2(W{2}))
    sprintf('standard deviation of W2 is %f',std2(W{2}))
%     sprintf('mean of b1 is %f',mean2(b{1}))
%     sprintf('standard deviation of b1 is %f',std2(b{1}))
end
end

function [trainRecord,validRecord]= Run2(trainP,validP,GDparams,W,b,lambda)
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
train_cost(1) = ComputeCost2(train_X,train_Y,Wstar,bstar, lambda);
train_acc(1)  = ComputeAccuracy2(train_X,train_y,Wstar,bstar);
valid_cost(1) = ComputeCost2(valid_X,valid_Y,Wstar,bstar, lambda);
valid_acc(1)  = ComputeAccuracy2(valid_X,valid_y,Wstar,bstar);

% run epoch
for i = 1:GDparams.n_epochs
    if rem(i,1)==0
         GDparams.eta = GDparams.eta*GDparams.decay_rate;
    end
    [Wstar, bstar] = MiniBatchGD2(train_X, train_Y, GDparams, Wstar, bstar, lambda);
    train_cost(i+1) = ComputeCost2(train_X,train_Y,Wstar,bstar, lambda);
    train_acc(i+1)  = ComputeAccuracy2(train_X,train_y,Wstar,bstar);
    valid_cost(i+1) = ComputeCost2(valid_X,valid_Y,Wstar,bstar, lambda);
    valid_acc(i+1)  = ComputeAccuracy2(valid_X,valid_y,Wstar,bstar);
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
    title(  ['lambda=',  num2str(lambda),...
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