function [alpha, w_k, tauk, obj_hist] = prox_gradient_lr(Data, y, p, sub_feat_ind, B, C, w_k, tau, eps_inner, max_iter, verbose)
% Master problem optimization w.r.t. the logistic loss in the primal form
%
%Input:
%   sub_feat_ind:       the indx of selected features, such as 
%   sub_feat_ind  = [0 10 15 n_feat], where n_feat is the number of
%   selected features
% Written by Mingkui Tan. If any comments, please email tanmingkui@gmail.com
%%
if nargin < 11
    verbose = 0;
end
if nargin < 10
    max_iter = 20;
end
if nargin < 9
    eps_inner = 0.001;
end

%%
%do initialization
t_1 = 1;
t_0 = 1;
yita = 0.6;
fw_temp = inf;

n_feat = sub_feat_ind(end);
SG = zeros(n_feat,1);
normsq = zeros(1,p);

%%
max_line_iter = 40;
obj_hist = [];   
w_k_1 = w_k;
for iter = 1:max_iter
    v = w_k + (t_1-1)/t_0*(w_k-w_k_1);
    [fv, dw] = calculate_fobj_gw(Data, y, v, p, B, C, sub_feat_ind);
  
    if (1==iter)
        for k=1:p
            normsq(k) = norm(w_k(sub_feat_ind(k)+1:sub_feat_ind(k+1)));
        end
        btot = sum(normsq)^2;
        fw_temp = fv + 0.5*btot;
        obj_hist(iter) = fw_temp;
    end
    %  begin the linear search
    tau = tau*yita;
    
    for t = 1:max_line_iter
        s = 1/tau;
        G  = v(1:n_feat,1) - s*full(dw');
        for k=1:p
            normsq(k) = norm(G(sub_feat_ind(k)+1:sub_feat_ind(k+1)));
        end
        b0 = normsq;
        b = proximity_L1squared(b0, s);
        btot = sum(b)^2;
        scale = zeros(1,p);
        ind = find(b0>0);
        scale(ind) = b(ind)./b0(ind);
        %     compute scaled SG after thresholding
        for k=1:p
            SG(sub_feat_ind(k)+1:sub_feat_ind(k+1)) = scale(k)*(G(sub_feat_ind(k)+1:sub_feat_ind(k+1)));
        end
    
        y_test = 0;
        for k=1:p
            v_ = SG(sub_feat_ind(k)+1:sub_feat_ind(k+1));
            x_data = Data{k};
            y_test = y_test + x_data*v_;
        end
        y_test = -y_test.*y; 
        alpha = exp(y_test);
        y_test_sq = log(1+alpha);
        fG_loss = C*sum(y_test_sq);
        fG = fG_loss + 0.5*btot;
    
        %    compoute fQ;
        fQ = fv + full(dw)*(SG-v(1:n_feat))+ 0.5*tau*norm(SG-v(1:n_feat))*norm(SG-v(1:n_feat)) + 0.5*btot;
        if fG<fQ
            tauk = tau;
        break
        else
            tau = tau/yita;
        end
    end

% update w_k, w_k_1
fw = fG;
obj_hist(iter+1) = fw;
w_k_1(1:n_feat) = w_k(1:n_feat);
w_k(1:n_feat) = SG(1:n_feat);
if(~isempty(obj_hist))
    f_stop = obj_hist(1);
else
    f_stop = fw_temp;
end
tmp = ((fw_temp - fw)<eps_inner*abs(f_stop)&&iter>1);
if tmp
    break;
end
fw_temp = fG;
t_1 = t_0;
t_0 = (1+sqrt(1+4*t_0^2))/2;
end
alpha = alpha./(1+alpha); 


if verbose
figure        
semilogy(obj_hist(1:iter+1),'DisplayName','obj_hist','YDataSource','obj_hist');figure(gcf)
end

btot = sum(b0);
betas = b0 / btot; % the value for mu in the dual problem
end

