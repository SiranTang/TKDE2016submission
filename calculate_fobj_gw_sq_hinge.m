function [f_obj, gw] = calculate_fobj_gw_sq_hinge(Data, y, v, p, B, C, sub_feat_ind)
% Calculate the objective value and gradient at the point v w.r.t the
% squred hinge loss
%   
%%
y_test = zeros(length(y), 1);

for k=1:p
    v_ = v(sub_feat_ind(k)+1:sub_feat_ind(k+1));
    x_data = Data{k};
    y_test = y_test + x_data*v_;
end
    %x_i = 1-y_i w'x
    y_test = 1 - y_test.*y; 
    
    %max(0, xi)
    y_test(y_test<0) = 0;
    
    %xi^2
    y_test_sq = y_test.*y_test;
    
    %y = \sum xi^2
    fv = sum(y_test_sq);
    f_obj = C*fv;
    y_test = y_test.*y;
    for k=1:p
        y_testy = repmat(y_test,1,size(Data{k},2)); 
        dw = y_testy.*Data{k};
        %dw = \sum dw_i
        dw_sum = sum(dw,1);
        gw(sub_feat_ind(k)+1:sub_feat_ind(k+1)) = -2*C*dw_sum;
    end
    
end
