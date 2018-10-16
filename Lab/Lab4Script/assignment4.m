%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a = W*h+U*x+b
% h = tanh(a)
% o = V*h+c
% p = softmax(o)
% W ---(m x m)
% U ---(m x K)
% v ---(k x m)
% h ---(m x 1)
% x ---(K x 1)
% b ---(m x 1)
% c ---(K x 1)
clc; close all; clear;

% read data
[ind_to_char,char_to_ind,book_data] = Read_Data('data/Goblet.txt');

% check map
Check_Map(ind_to_char,char_to_ind);

% init
[GDparam,RNN] = ParamInit(ind_to_char);

% synthesize text
n =10;
x_0 = 'b';
h = zeros(GDparam.m,1);
[generated_onehot,generated_txt] = txt_generator(n,h,GDparam,x_0,RNN, char_to_ind,ind_to_char);

% calculate gradient
%
X_chars = book_data(1:GDparam.seq_length);
Y_chars = book_data(2:GDparam.seq_length+1);
X_trans = to_onehot(X_chars,char_to_ind);
Y_trans = to_onehot(Y_chars,char_to_ind);

h0 = zeros(GDparam.m,1);
l1 = ComputeLoss(X_trans, Y_trans, RNN, h0);
[a,h,o,p] = Evaluatesynth(X_trans,h0,RNN);
grads = ComputeGradients(X_trans,Y_trans,RNN,a,h,p);
num_grads = ComputeGradsNum(X_trans, Y_trans, RNN, 1e-4);
f = fieldnames(grads)';
for i=1:numel(f)
  diff.(f{i}) = norm(grads.(f{i})-num_grads.(f{i}))/max([1e-6,norm(grads.(f{i}))+norm(num_grads.(f{i}))]);
  sprintf('the difference of gradient %s between two method is %f',(f{i}),diff.(f{i}))
end
%}
% run sgd
%
[GDparam,RNN] = ParamInit(ind_to_char);
GDparam.epochnum = 20;
smooth_box = MiniBatchGD(RNN,GDparam);
save('smooth_box.mat','smooth_box');
%}
% plot loss
%
smooth_box = load('smooth_box.mat');
figure;
size(smooth_box.smooth_box)
plot(smooth_box.smooth_box)
% hold on 
% legend('train cost','valid cost')
xlabel('iteration')
ylabel('loss')
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% functions
function  [ind_to_char,char_to_ind,book_data] = Read_Data(book_fname)
% book_fname = 'data/Goblet.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);
book_chars = unique(book_data);
char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');
char_len = size(book_chars,2);
for i = 1:char_len
    char_to_ind(book_chars(i)) = i;
    ind_to_char(i) = book_chars(i);
end
end

function Check_Map(ind_to_char,char_to_ind)
 k = size(ind_to_char,1);
 for i = 1:k
     if char_to_ind(ind_to_char(i)) ~= i 
         sprintf('There is smothing wrong with Mapping !!')
         break  
     end
 end
 sprintf('good !')

end

function [a_t,h_t,o_t,p_t] = synthesize(x_t,h_t,RNN)
a_t = RNN.W*h_t + RNN.U*x_t + RNN.b;
h_t = tanh(a_t);
o_t = RNN.V*h_t + RNN.c;
p_t = exp(o_t)./repmat(sum(exp(o_t)),size(o_t,1),1);
end

function transfered = to_onehot(data,char_to_ind)
   K = size(char_to_ind,1);
   transfered = zeros(K,size(data,2));
   for i = 1:size(data,2)
       transfered(char_to_ind(data(i)),i) = 1;
   end
end

function [generated_onehot,generated_txt] = txt_generator(n,h,GDparam,x_0, RNN,char_to_ind,ind_to_char)
h_t = h;
x_t= to_onehot(x_0,char_to_ind);
generated_onehot = zeros(GDparam.K,n);
generated_txt = '';
for i = 1:n 
    [~,h_t,~,p_t] = synthesize(x_t,h_t,RNN);
    cp = cumsum(p_t);
    a = rand;
    ixs = find(cp-a>0);
    ii = ixs(1);
    x_t = zeros(GDparam.K,1);
    x_t(ii,1) = 1;
%     generated_txt = strcat(generated_txt,ind_to_char(ii));
    generated_txt = [generated_txt,ind_to_char(ii)];
    generated_onehot(:,i) = x_t;
end
end

function [a,h,o,p] = Evaluatesynth(X,h0,RNN)
n = size(X,2);
m = size(RNN.W,1);
k = size(RNN.c,1);
a = zeros(m,n);
h = zeros(m,n+1);
o = zeros(k,n);
p = zeros(k,n);
h_t = h0;
h(:,1) = h0;
for i = 1:n
    [a_t,h_t,o_t,p_t] = synthesize(X(:,i),h_t,RNN);
    h(:,i+1) = h_t;
    a(:,i) = a_t;
    o(:,i) = o_t;
    p(:,i) = p_t;
end
end

function l1 = ComputeLoss(X, Y, RNN_try, hprev)
    l1 = 0;
    h = hprev;
    n = size(X,2);
    for i = 1:n
        [~,h,~,p_t] = synthesize(X(:,i),h,RNN_try);
        l1 = l1 - log(Y(:, i)'*p_t);
    end
end

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

function [GDparam,RNN] = ParamInit(ind_to_char)
rngnumber = 40;
rng(rngnumber)
m = 100;
K = size(ind_to_char,1);

GDparam.m = m;
GDparam.K = K;
GDparam.eta = 0.1;
GDparam.epochnum = 30;
GDparam.seq_length = 25;

sig = 0.01;
RNN.b = zeros(m,1);
RNN.c = zeros(K,1);
RNN.U = randn(m, K)*sig;
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;
end

function  smooth_box = MiniBatchGD(RNN,GDparam)

[ind_to_char,char_to_ind,book_data] = Read_Data('data/Goblet.txt');
epoch = 1;
smooth_box = [];
e = 1;
iter = 1;
f = fieldnames(RNN)';
for i=1:numel(f)
  Mthe.(f{i}) = zeros(size(RNN.(f{i})));
end
while epoch <= GDparam.epochnum 
       
    X_chars = book_data(e:e+GDparam.seq_length-1);
    Y_chars = book_data(e+1:e+GDparam.seq_length);
    X_trans = to_onehot(X_chars,char_to_ind);
    Y_trans = to_onehot(Y_chars,char_to_ind);
    if e == 1 && iter ==1
        h0 = zeros(GDparam.m,1);
    else
        h0 = h(:,end);
    end
    [a,h,~,p] = Evaluatesynth(X_trans,h0,RNN);
    ce = ComputeLoss(X_trans, Y_trans, RNN, h0);
    grads = ComputeGradients(X_trans,Y_trans,RNN,a,h,p);
    %%%%%%
    if epoch == 1&&e ==1
        smooth_loss = ce;
        smooth_box = [smooth_box,smooth_loss];
        sprintf('smooth_loss: %f',smooth_loss)
        save('smooth_box.mat','smooth_box');
    else
        smooth_loss = 0.999* smooth_loss + 0.001 * ce;
        smooth_box = [smooth_box,smooth_loss];
    end
    
    %f = fieldnames(RNN)';
    % Adagrad
    %for i=1:numel(f) 
    for f = fieldnames(grads)'
      % clip gradient
      grads.(f{1}) = max(min(grads.(f{1}), 10), -10);
      Mthe.(f{1}) = Mthe.(f{1})+grads.(f{1}).^2;
      RNN.(f{1}) = RNN.(f{1})-GDparam.eta*grads.(f{1})./sqrt(Mthe.(f{1})+1e-9);
    end
   % show loss
%     if rem(iter,100) == 0
%         smooth_box = [smooth_box,smooth_loss];
%         save('smooth_box.mat','smooth_box');
%     end
    % show txt
    if rem(iter,10000) == 0 || iter ==1
        n = 1000;
        x_0 = X_chars(:,1);
        sprintf('smooth_loss: %f',smooth_loss)  
        save('smooth_box.mat','smooth_box');
        [~,generated_txt] = txt_generator(n,h0,GDparam,x_0, RNN,char_to_ind,ind_to_char);
        sprintf('-------- epoch %d iterataion %d -------------',epoch,iter)
%         sprintf('generated txt :\n')
        disp(generated_txt);

    end

    e = e+GDparam.seq_length;
    iter = iter +1;
    if  e> length(book_data)-GDparam.seq_length-1
        epoch = epoch +1;
        e = 1; 
        save('RNN.mat','RNN');
    end

end

end

function num_grads = ComputeGradsNum(X, Y, RNN, h)

for f = fieldnames(RNN)'
    disp('Computing numerical gradient for')
    disp(['Field name: ' f{1} ]);
    num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h);
end
end
function grad = ComputeGradNumSlow(X, Y, f, RNN, h)

n = numel(RNN.(f));
grad = zeros(size(RNN.(f)));
hprev = zeros(size(RNN.W, 1), 1);
for i=1:n
    RNN_try = RNN;
    RNN_try.(f)(i) = RNN.(f)(i) - h;
    l1 = ComputeLoss(X, Y, RNN_try, hprev);
    RNN_try.(f)(i) = RNN.(f)(i) + h;
    l2 = ComputeLoss(X, Y, RNN_try, hprev);
    grad(i) = (l2-l1)/(2*h);
end
end



