%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;
%% load training, validation and test data
% X (d x N) d:dimensionality of each image N:image number
% y (N x 1) is the vector label of images 
% Y (k x N) k is class number
[train_X, train_Y, train_y] = LoadBatch('data_batch_1.mat');
[valid_X, valid_Y, valid_y] = LoadBatch('data_batch_2.mat');
[test_X,  test_Y,  test_y ] = LoadBatch('test_batch.mat');
% show the images of train_X
I = reshape(train_X, 32, 32, 3, 10000);
I = permute(I, [2, 1, 3, 4]);
%montage(I(:, :, :, 1:500), 'Size', [5,5]);

%% initialize W(K x d) and b(K X 1)
rng(1);
mu = 0;
sigma = 0.01;
W = mu + sigma*randn(size(train_Y,1),size(train_X,1),'double');
b = mu + sigma*randn(size(train_Y,1),1,'double');
% check answer
sprintf('mean of W is %f',mean2(W))
sprintf('standard deviation of W is %f',std2(W))
sprintf('mean of b is %f',mean2(b))
sprintf('standard deviation of b is %f',std2(b))

%% check P of Classifier
P = EvaluateClassifier(train_X(:, 1:100), W, b);

%% cost and accuracy of Classifier
lambda = 0;
train_cost = ComputeCost(train_X,train_Y,W,b, lambda);
sprintf('the train_cost is %f',train_cost)
train_acc = ComputeAccuracy(train_X,train_y,W,b);
sprintf('the train_acc is %f',train_acc)

%% Calculate gradient of Classifier
[grad_W, grad_b] = ComputeGradients(train_X(:, 1), train_Y(:,1), P, W, lambda);
% compare gradient 
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(train_X(:, 1),train_Y(:,1), W, b, lambda, 1e-6);
diff_W = norm(grad_W-ngrad_W)/max([1e-6,norm(grad_W)+norm(ngrad_W)]);
sprintf('the difference of gradient W between two method is %f',diff_W)
diff_b = norm(grad_b-ngrad_b)/max([1e-6,norm(grad_b)+norm(ngrad_b)]);
sprintf('the difference of gradient b between two method is %f',diff_b)

%% mini-batch gradient descent
GDparams.n_batch = 100;
GDparams.eta = 0.01;
GDparams.n_epochs = 40;
lambda = 0;
train_cost = zeros(GDparams.n_epochs,1);
train_acc  = zeros(GDparams.n_epochs,1);
valid_cost = zeros(GDparams.n_epochs,1);
valid_acc  = zeros(GDparams.n_epochs,1);
Wstar = W;
bstar = b;
for i = 1:GDparams.n_epochs
[Wstar, bstar] = MiniBatchGD(train_X, train_Y, GDparams, Wstar, bstar, lambda);
train_cost(i) = ComputeCost(train_X,train_Y,Wstar,bstar, lambda);
train_acc(i)  = ComputeAccuracy(train_X,train_y,Wstar,bstar);
valid_cost(i) = ComputeCost(valid_X,valid_Y,Wstar,bstar, lambda);
valid_acc(i)  = ComputeAccuracy(valid_X,valid_y,Wstar,bstar);
end

%% plot cost
figure(1)
plot(train_cost)
hold on 
plot(valid_cost)
legend('train cost','valid cost')
xlabel('epoch')
ylabel('cost')
%% plot W
for i=1:10
im = reshape(Wstar(i, :), 32, 32, 3);
s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure(2)
montage(s_im, 'Size', [2,5]);

%%  Run different parameter settings
% lambda = 0, n_epochs = 40, n_batch = 100, eta = .1
GDparams = GDparamSetting(40,100,0.1);
Run(train_X,train_Y,train_y,valid_X,valid_Y,valid_y,GDparams,W,b,0,true);
% lambda = 0, n_epochs = 40, n_batch = 100, eta = .01
GDparams = GDparamSetting(40,100,0.01);
Run(train_X,train_Y,train_y,valid_X,valid_Y,valid_y,GDparams,W,b,0,true);
% lambda = 0.1, n_epochs = 40, n_batch = 100, eta = .01
GDparams = GDparamSetting(40,100,0.01);
Run(train_X,train_Y,train_y,valid_X,valid_Y,valid_y,GDparams,W,b,0.1,true);
% lambda = 1, n_epochs = 40, n_batch = 100, eta = .01
GDparams = GDparamSetting(40,100,0.01);
Run(train_X,train_Y,train_y,valid_X,valid_Y,valid_y,GDparams,W,b,1,true);

%%  Test
% lambda = 0, n_epochs = 40, n_batch = 100, eta = .1
GDparams = GDparamSetting(40,100,0.1);
[train_cost1,train_acc1,test_cost1,test_acc1] = ...
Run(train_X,train_Y,train_y,test_X,test_Y,test_y,GDparams,W,b,0,false);
% lambda = 0, n_epochs = 40, n_batch = 100, eta = .01
GDparams = GDparamSetting(40,100,0.01);
[train_cost2,train_acc2,test_cost2,test_acc2] = ...
Run(train_X,train_Y,train_y,test_X,test_Y,test_y,GDparams,W,b,0,false);
% lambda = 0.1, n_epochs = 40, n_batch = 100, eta = .01
GDparams = GDparamSetting(40,100,0.01);
[train_cost3,train_acc3,test_cost3,test_acc3] = ...
Run(train_X,train_Y,train_y,test_X,test_Y,test_y,GDparams,W,b,0.1,false);
% lambda = 1, n_epochs = 40, n_batch = 100, eta = .01
GDparams = GDparamSetting(40,100,0.01);
[train_cost4,train_acc4,test_cost4,test_acc4] = ...
Run(train_X,train_Y,train_y,test_X,test_Y,test_y,GDparams,W,b,1,false);
%create table for testing
testname = {'test1';'test2';'test3';'test4'};
lambda   = [0;0;0.1;1];
n_epochs = [40;40;40;40];
n_batch  = [100;100;100;100];
eta      = [0.1;0.01;0.01;0.01];
train_accuracy = [train_acc1(end);train_acc2(end);train_acc3(end);train_acc4(end)];
test_accuracy  = [test_acc1(end);test_acc2(end);test_acc3(end);test_acc4(end)];
T = table(testname,lambda,n_epochs,n_batch,eta,train_accuracy,test_accuracy)

%% Functions
function[X,Y,y] = LoadBatch(filename)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Imagedata = load(filename);
% X (d x N) d:dimensionality of each image N:image number
X = double(Imagedata.data')./255;
% y (N x 1) is the vector label of images 
y = Imagedata.labels+1;
% Y (k x N) k is class number
Y = zeros(length(unique(y)),length(y));
for i = 1: length(y)
    Y(y(i),i) = 1;
end
end

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
end

function J = ComputeCost(X,Y,W,b, lambda)
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
P = EvaluateClassifier(X,W,b);
D = size(X,2);
% Y'P ---> (1 x K) x (K x 1) if Y is a vector
% Y.*p for extracting the matched P if Y is a matrix. 
J  = -1/D*sum(log(sum(Y.*P)))+lambda*sum(sum(W.^2));
end

function acc = ComputeAccuracy(X,y,W,b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X (d x N)
% y (N x 1)
% W (K x d)
% b (K x 1)

% method 1
P = EvaluateClassifier(X,W,b);
[~,Index] = max(P);
acc = sum(Index == y')/length(y);
% method 2
% use onehot Y .* onehot outcome and then sum up
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X (d x N)
% Y (K x N)
% P (K x N)
% W (K x d)
% drad_w (K x d)
% grad_b (K x 1)
N = size(Y,2);
[K,~] = size(W);
[d,~] = size(X);
grad_W = zeros(K,d);
grad_b = zeros(K,1);
% calculate g following the lecture 3
for i = 1:N
    Yn = Y(:,i);
    Pn = P(:,i);
    Xn = X(:,i);
    % (1 x k)/(1 x k x k x k) x [(K x K)- K x 1 x 1 x K]= 1 x K
    % so the size of g ( 1 x K )
    g = - Yn'/(Yn'*Pn)*(diag(Pn)-Pn*Pn');
    % check whether g == g2
%     g2 = -(Yn-Pn)';
%     if g ~= g2
%         disp('g and g2 not equal!');
%         break;
%     end
    % gradient J w.r.t bias = g
    grad_b = grad_b +g';
    % gradient J w.r.t W = g'x 
    grad_W = grad_W +g'*Xn';
end

grad_W = (1/N)*grad_W+ 2*lambda*W;
grad_b = (1/N)*grad_b;

end

function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)
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
eta = GDparams.eta;
% n_epochs = GDparams.n_epochs;
Wstar = W; 
bstar = b;
Xtrain = X;
Ytrain = Y;
% run epoch
% for i = 1:n_epochs
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = Xtrain(:, inds);
        Ybatch = Ytrain(:, inds);
        P = EvaluateClassifier(Xbatch,Wstar,bstar);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, Wstar, lambda);
        Wstar = Wstar - eta*grad_W;
        bstar = bstar - eta*grad_b;
    end
% end
end

function [train_cost,train_acc,valid_cost,valid_acc]= Run(train_X,train_Y,train_y,valid_X,valid_Y,valid_y,GDparams,W,b,lambda,ifplot)

% set cost and acc
train_cost = zeros(GDparams.n_epochs,1);
train_acc  = zeros(GDparams.n_epochs,1);
valid_cost = zeros(GDparams.n_epochs,1);
valid_acc  = zeros(GDparams.n_epochs,1);
Wstar = W;
bstar = b;

% run epoch
for i = 1:GDparams.n_epochs
[Wstar, bstar] = MiniBatchGD(train_X, train_Y, GDparams, Wstar, bstar, lambda);
train_cost(i) = ComputeCost(train_X,train_Y,Wstar,bstar, lambda);
train_acc(i)  = ComputeAccuracy(train_X,train_y,Wstar,bstar);
valid_cost(i) = ComputeCost(valid_X,valid_Y,Wstar,bstar, lambda);
valid_acc(i)  = ComputeAccuracy(valid_X,valid_y,Wstar,bstar);
end
if ifplot
    % plot cost
    figure
    plot(train_cost)
    hold on 
    plot(valid_cost)
    legend('train cost','valid cost')
    xlabel('epoch')
    ylabel('cost')
    title(  ['lambda=',  num2str(lambda),...
            'epochs=',num2str(GDparams.n_epochs),...
            'batch=', num2str(GDparams.n_batch),...
            'eta=',     num2str(GDparams.eta)]);
    % plot W
    for i=1:10
    im = reshape(Wstar(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
    end
    figure
    title(  ['lambda=',  num2str(lambda),...
            'epochs=',num2str(GDparams.n_epochs),...
            'batch=', num2str(GDparams.n_batch),...
            'eta=',     num2str(GDparams.eta)]);
    montage(s_im, 'Size', [2,5]);
end

end

function GDparams = GDparamSetting(n_epochs,n_batch,eta)
GDparams.n_batch = n_batch;
GDparams.eta = eta;
GDparams.n_epochs = n_epochs;
end

%% Given functions (except montage.m)
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    c1 = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    c1 = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end

end

function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

c = ComputeCost(X, Y, W, b, lambda);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c) / h;
end

for i=1:numel(W)   
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c) / h;
end
end


