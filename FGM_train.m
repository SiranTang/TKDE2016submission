function [w, feat_ind, obj_hist, obj_hist_out] = FGM_train(X, y, B, C, loss_type, group_ind, eps, max_iter, verbose)
%   [w train_loss] = FGM_train(X, y, B, C, max_iter, eps) is the main function of
%   "Feature Generating Machine (FGM)", which chooses a set of features to
%   represent the original data X
%
%--------------------------------------------------------------------------
%Input:
%   X:          n by m data matrix with n instances and m features
%   y:          n by 1
%   B:          least number of features to be selected (default value: 5)
%   C:          penalty parameter (default value: 10)
%   loss_type:  loss type used in training. There are two options now,
%               namely 'squred_hinge' and 'logistic' (default value: 10)
%   group_ind:  indicator for group information. If group==0, there is no
%               group information involved (default value: 0 for linear 
%               feature selection)
%   eps:        stopping tolerance (default value: 0.001)
%   max_iter:   maximum of iterations (default value: 20)
%   verbose:    whether plot the history of objectives or not ((default value: 0)
%
%--------------------------------------------------------------------------
%Output:
%   w:              an n_feat by 1 weight vector for the selected features
%   feat_ind:       indices of the selected features
%   alpha:          a n by 1 vector to represent the training error for each
%                   instance
%   obj_hist:       history of objective values for all PG iterations
%   obj_hist_out:   history of objective values for FGM outer iterations
%
%--------------------------------------------------------------------------
%Details for "group_ind" construction when doing group feature selection: 
%   The variable "group" records the ending index per group. Notice that, 
%   the first element is 0, the end element is the total number of features.
%   For instance, if the features are grouped as {{1,2}, {3,4,5}, ....{m-2,
%   m-1, m}}, group_ind should be [0, 2, 5, ..., m].
%
%--------------------------------------------------------------------------
%Traing usage:
%    [w_train, feat_ind, obj_hist, obj_hist_out] = FGM_train(X, y, B, C,
%    loss_type);
%
%--------------------------------------------------------------------------
%Predction usage:
%    [y_predict] = X_test(:,feat_ind)*w_train;
%    y_diff = sign(y_predict)-y_test;
%    error_rate = nnz(y_diff)/size(X_test,1);
%
%--------------------------------------------------------------------------
%Toy example:
% n = 4096; %number of training instances
% n_t = 2000; %number of testing instances
% m = 10000; %number of features
% w = 5*randn(m,1);
% w(1000:10000) = 0;
% X_train = randn(n,m);
% y = sign(X_train*w);
% X_test = randn(n_t,m);
% y_test = sign(X_test*w);
% group_ind = [0:5:1000];
%
%--------------------------------------------------------------------------
% B = 5;
% C = 5;
% loss_type = 'squred_hinge';
% [w_train, feat_ind] = FGM_train(X_train, y, B, C, loss_type, group_ind);
% [y_predict] = X_test(:,feat_ind)*w_train;
% y_diff = sign(y_predict)-y_test;
% error_rate = nnz(y_diff)/size(X_test,1);
%
%Written by Mingkui Tan. Any comments or bugs please email tanmingkui@gmail.com
if nargin < 9
    verbose = 0;
end 
if nargin < 8
    max_iter = 20;
end
if nargin < 7
    eps = 0.001;
end
if nargin < 6
    group_ind = 0; 
end
if nargin < 5
    loss_type = 'squred_hinge'; 
end
if nargin < 4
    C = 10;
end
if nargin < 3
    B = 5;
end

% Initialization
[n, m] = size(X);
alpha = ones(n,1);

[d_indx] = most_violated_analysis(X, y, alpha, B, group_ind);
sub_feat_ind = [0 length(d_indx)];
%the length of d_indx may be different, thus I use cell structure
D_indx{1} = d_indx;
Data{1} = X(:,d_indx);
w_k = zeros(m,1);
w_k1 = zeros(m,1);


tau = 0.1*n*C;
obj_hist_out = zeros(max_iter+1,1);
obj_hist = [];
iter = 1;
while(iter<max_iter)
    if strcmp(loss_type,'squred_hinge')
        [alpha, w_k, tau, f_master_hist] = prox_gradient_sq_hinge(Data, y, iter, sub_feat_ind, B, C, w_k, tau);
    end
    if strcmp(loss_type,'logistic')
        [alpha, w_k, tau, f_master_hist] = prox_gradient_lr(Data, y, iter, sub_feat_ind, B, C, w_k, tau);
    end
    tau = 0.5*tau;
    if 1==iter
    obj_hist_out(iter) = f_master_hist(1);
    end
    obj_hist_out(iter+1) = f_master_hist(end);
    obj_hist = [obj_hist f_master_hist];

    if (iter>2)
        diff = abs(obj_hist_out(iter)  - obj_hist_out(iter-1));
        if diff < obj_hist_out(1)*eps
            break;
        end
    end
    [d_indx] = most_violated_analysis(X, y, alpha, B, group_ind);
    sub_feat_ind = [sub_feat_ind sub_feat_ind(end)+length(d_indx)];
    D_indx{iter+1} = d_indx;
    Data{iter+1} = X(:,d_indx);
    iter = iter + 1;
end

%%
%there may be some overlapping features. However, this does not affact the prediction.  
feat_ind = [];
for i=1:iter
    feat_ind = [feat_ind D_indx{i}];
end
[ind, order] = sort(feat_ind);
w = w_k(order);
feat_ind = ind;


if verbose
figure        
semilogy(obj_hist_out(1:iter),'DisplayName','obj_hist_out','YDataSource','obj_hist');figure(gcf)
figure        
semilogy(obj_hist,'DisplayName','obj_hist_all_PG','YDataSource','obj_hist');figure(gcf)
end

end

