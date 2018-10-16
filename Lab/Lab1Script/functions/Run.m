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