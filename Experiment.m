function  Experiment(source_domains, target_domain)
% Experiment: the main function used to run HomOTLMS.
% 
%--------------------------------------------------------------------------
% Input:
%      source_domains: a set of source dataset_name, e.g. {'FR-EN_GR-EN', 
%                      'GR-EN_IT-EN', 'IT-EN_SP-EN', 'SP-EN_EN-EN'}
%      target_domain: target dataset_name, e.g. 'EN-EN_FR-EN'
%--------------------------------------------------------------------------


%load dataset
train_data = [];
for i = 1:length(source_domains),
    load(sprintf('data/%s', source_domains{i}));
    train_data = [train_data; data(ID_old, :)];
end

[train_n, train_d] = size(train_data);
X_train = train_data(:, 2:train_d);
y = train_data(:, 1);

B = 5;
C = 5;
loss_type = 'squred_hinge';
[w_train, feat_ind] = FGM_train(X_train, y, B, C, loss_type);
feat_ind = unique(feat_ind) + 1;

train_data = [];
for i = 1:length(source_domains),
    load(sprintf('data/%s', source_domains{i}));
    train_data = [train_data; [data(ID_old, 1), data(ID_old, feat_ind)]];
end

load(sprintf('data/%s', target_domain));
test_data = [data(ID_old, 1), data(ID_old, feat_ind)];

data = [train_data; test_data];
[n,d]       = size(data);
NumOld = size(train_data, 1);
[ID_old, ID_new] = create_OTL_ID(NumOld, n);

% options
options.C      = 5;
options.t_tick = round(length(ID_new)/15);

%
m = length(ID_new);
options.beta1 = sqrt(m)/(sqrt(m)+sqrt(log(2)));
options.beta2 = sqrt(m)/(sqrt(m)+sqrt(log(length(source_domains) + 1)));
options.Number_old=n-m;
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

X2 = X(n-m+1:n,:);
Y2 = Y(n-m+1:n);


%% run experiments:
    
% learn the old classifier
[h, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = avePA1_K_M(Y, X, options, ID_old);

hs = [];
for i = 1:length(source_domains)
    hs = [hs, source_classifier(source_domains{i}, options, feat_ind)];
end

for i=1:size(ID_new,1),
    fprintf(1,'running on the %d-th trial...\n',i);
    ID = ID_new(i, :);
    
    fcsv = fopen('trials.csv', 'a+');
    fprintf(fcsv, '%d-th trial, accuracy, precision, recall, F_measure, MCC, \n', i);
    
    % HomOTLMS
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs, accuracy, precision, recall, F_measure, MCC, accu, prec, rec, fm, mcc] = HomOTLMS(Y,X,X2,options,ID,hs);
    nSV_OTLMS(i) = classifier.numSV;
    err_OTLMS(i) = err_count;
    time_OTLMS(i) = run_time;
    mistakes_list_OTLMS(i,:) = mistakes;
    TMs_OTLMS(i,:) = TMs;
    accuracy_OTLMS(i,:) = accuracy;
    precision_OTLMS(i,:) = precision;
    recall_OTLMS(i,:) = recall;
    F_measure_OTLMS(i,:) = F_measure;
    MCC_OTLMS(i,:) = MCC;
    accu_OTLMS(i,:) = accu;
    prec_OTLMS(i,:) = prec;
    rec_OTLMS(i,:) = rec;
    fm_OTLMS(i,:) = fm;
    mccs_OTLMS(i,:) = mcc;
    fprintf(fcsv, 'HomOTLMS, %.4f, %.4f, %.4f, %.4f, %.4f,\n', accuracy, precision, recall, F_measure, MCC);
    
    fclose(fcsv);
end

fprintf(1,'-------------------------------------------------------------------------------\n');
fprintf(1,'number of mistakes,            size of support vectors,           cpu running time\n');
fprintf(1,'HomOTLMS       %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_OTLMS)/m*100,   std(err_OTLMS)/m*100, mean(nSV_OTLMS), std(nSV_OTLMS), mean(time_OTLMS), std(time_OTLMS));
fprintf(1,'-------------------------------------------------------------------------------\n');

