function [ h ] = source_classifier(dataset_name, options, feat_ind)

%load dataset
load(sprintf('data/%s', dataset_name));
[n,d]       = size(data);

data = [data(ID_old, 1), data(ID_old, feat_ind)];
[n,d]       = size(data);

Y=data(1:n,1);
Y=full(Y);
X = data(1:n,2:d);


% scale
MaxX=max(X,[],2);
MinX=min(X,[],2);
DifX=MaxX-MinX;
idx_DifNonZero=(DifX~=0);
DifX_2=ones(size(DifX));
DifX_2(idx_DifNonZero,:)=DifX(idx_DifNonZero,:);
X = bsxfun(@minus, X, MinX);
X = bsxfun(@rdivide, X , DifX_2);

[h, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = avePA1_K_M(Y, X, options, ID_old);

end

