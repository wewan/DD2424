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