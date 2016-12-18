function [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = avePA1_K_M(Y, X, options, id_list)
% avePA1: Averge online passive-aggressive algorithm
%--------------------------------------------------------------------------
% Input:
%        Y:    the vector of lables
%        K:    precomputed kernel for all the example, i.e., K_{ij}=K(x_i,x_j)
%  id_list:    a randomized ID list
%  options:    a struct containing rho, sigma, C, n_label and n_tick;
% Output:
%   err_count:  total number of training errors
%    run_time:  time consumed by this algorithm once
%    mistakes:  a vector of mistake rate 
% mistake_idx:  a vector of number, in which every number corresponds to a
%               mistake rate in the vector above
%         SVs:  a vector records the number of support vectors 
%     size_SV:  the size of final support set
%--------------------------------------------------------------------------

%% initialize parameters
C = options.C; % 1 by default
t_tick = options.t_tick;
w = zeros(1, size(X, 2));
ave_w = zeros(1, size(X, 2));
SV = [];
ID = id_list;
err_count = 0;
mistakes = [];
mistakes_idx = [];
SVs = [];
TMs=[];
%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    
    x_t = X(id, :);
    f_t = w*x_t';            % decision function

    l_t = max(0,1-Y(id)*f_t);   % hinge loss
    hat_y_t = sign(f_t);        % prediction
    if (hat_y_t==0)
        hat_y_t=1;
    end
    % count accumulative mistakes
    if (hat_y_t~=Y(id)),
        err_count = err_count + 1;
    end
    
    if (l_t>0)
        % update 
        s_t = norm(x_t)^2;
        gamma_t = min(C,l_t/s_t);
        
        w = w + gamma_t*Y(id)*x_t;
        
        ave_w = ((t-1)/t)*ave_w + (1/t)*w;
    else
        ave_w = ((t-1)/t)*ave_w + (1/t)*w;
    end
    
    
    run_time=toc;
    if (mod(t,t_tick)==0)
        mistakes = [mistakes err_count/t];
        mistakes_idx = [mistakes_idx t];
        SVs = [SVs length(SV)];
        TMs=[TMs run_time];
    end
end
classifier.w = ave_w;
fprintf(1,'The number of mistakes = %d\n', err_count);
run_time = toc;
