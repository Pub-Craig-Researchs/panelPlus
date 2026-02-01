function est = panelDDML(id, time, y, D, X, varargin)
%PANELDDML Double/Debiased Machine Learning for panel causal effects
%   Estimates causal effects using cross-fitting with ML first-stage.
%
%   est = PANELDDML(id, time, y, D, X) estimates ATE of D on y controlling for X
%   est = PANELDDML(id, time, y, D, X, Name, Value) with options:
%
%   Options:
%   - 'method': ML method for nuisance estimation
%       'lasso' - LASSO regression (default)
%       'ridge' - Ridge regression
%       'elasticnet' - Elastic net
%       'rf' - Random Forest (requires Statistics Toolbox)
%   - 'nFolds': Number of cross-fitting folds (default: 5)
%   - 'panelFE': Include individual fixed effects (default: true)
%   - 'timeFE': Include time fixed effects (default: true)
%   - 'lambda': Regularization parameter (default: auto CV)
%   - 'cluster': Cluster variable for SE (default: id)
%
%   The DML procedure:
%   1. Residualize Y on X: Y_tilde = Y - E[Y|X]
%   2. Residualize D on X: D_tilde = D - E[D|X]
%   3. Regress Y_tilde on D_tilde to get causal effect
%   Cross-fitting ensures valid inference.
%
%   Example:
%       est = panelplus.ml.panelDDML(id, time, outcome, treatment, controls, ...
%                   'method', 'lasso', 'nFolds', 5);
%
%   References:
%   - Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning
%   - Belloni, A., Chernozhukov, V., Hansen, C. (2014). High-Dimensional Methods
%
%   See also CAUSALFOREST
%
%   Copyright 2026 PanelPlus Toolbox

arguments
    id (:,1)
    time (:,1)
    y (:,1) double
    D (:,1) double
    X (:,:) double
end
arguments (Repeating)
    varargin
end

% Parse options
p = inputParser;
addParameter(p, 'method', 'lasso', @(x) ismember(x, {'lasso', 'ridge', 'elasticnet', 'rf'}));
addParameter(p, 'nFolds', 5, @(x) isnumeric(x) && x >= 2);
addParameter(p, 'panelFE', true, @islogical);
addParameter(p, 'timeFE', true, @islogical);
addParameter(p, 'lambda', [], @isnumeric);
addParameter(p, 'cluster', id, @(x) length(x) == length(y));
parse(p, varargin{:});
opts = p.Results;

% Initialize result
est = panelplus.utils.panelResult();
est.method = sprintf("Panel DDML (%s)", opts.method);
est.options = opts;

% Store original data
est.y = y;
est.X = X;

% Get panel structure
[uid, ~, idIdx] = unique(id);
[utime, ~, timeIdx] = unique(time);
n = length(uid);
T = length(utime);
N = length(y);
k = size(X, 2);

est.n = n;
est.T = T;
est.N = N;
est.k = k + 1;  % +1 for treatment

% Apply fixed effects transformations
if opts.panelFE
    y = withinTransform(y, idIdx, n);
    D = withinTransform(D, idIdx, n);
    for j = 1:k
        X(:, j) = withinTransform(X(:, j), idIdx, n);
    end
end

if opts.timeFE
    y = withinTransformTime(y, timeIdx, T);
    D = withinTransformTime(D, timeIdx, T);
    for j = 1:k
        X(:, j) = withinTransformTime(X(:, j), timeIdx, T);
    end
end

% Cross-fitting
nFolds = opts.nFolds;

% Create fold indices (stratified by individual if possible)
foldIdx = createFolds(idIdx, n, nFolds);

% Initialize residuals
yResid = zeros(N, 1);
dResid = zeros(N, 1);

% Cross-fitting loop
for fold = 1:nFolds
    testIdx = (foldIdx == fold);
    trainIdx = ~testIdx;

    XTrain = X(trainIdx, :);
    yTrain = y(trainIdx);
    DTrain = D(trainIdx);

    XTest = X(testIdx, :);
    yTest = y(testIdx);
    DTest = D(testIdx);

    % Fit ML model for E[Y|X]
    yHat = mlPredict(XTrain, yTrain, XTest, opts);

    % Fit ML model for E[D|X]
    dHat = mlPredict(XTrain, DTrain, XTest, opts);

    % Store residuals
    yResid(testIdx) = yTest - yHat;
    dResid(testIdx) = DTest - dHat;
end

% Final stage: regress Y_resid on D_resid
% theta = cov(Y_resid, D_resid) / var(D_resid)
theta = (dResid' * yResid) / (dResid' * dResid);

% Compute residuals from final stage
res = yResid - theta * dResid;

% Clustered standard error
varTheta = computeClusteredVarScalar(dResid, res, opts.cluster, N);
seTheta = sqrt(varTheta);

% t-statistic and p-value
tstat = theta / seTheta;
pval = 2 * (1 - normcdf(abs(tstat)));

% Store results
est.coef = theta;
est.varcoef = varTheta;
est.stderr = seTheta;
est.tstat = tstat;
est.pval = pval;
est.res = res;
est.k = 1;

% Method details
est.methodDetails.mlMethod = opts.method;
est.methodDetails.nFolds = nFolds;
est.methodDetails.yResid = yResid;
est.methodDetails.dResid = dResid;
est.methodDetails.r2_Y = 1 - var(yResid) / var(y);
est.methodDetails.r2_D = 1 - var(dResid) / var(D);

% Diagnostics
est.diagnostics.info = {
    sprintf('R² for Y model: %.3f', est.methodDetails.r2_Y);
    sprintf('R² for D model: %.3f', est.methodDetails.r2_D)
    };

est.isLinear = true;
est.isAsymptotic = true;
end

%% Helper Functions

function yW = withinTransform(y, idIdx, n)
%WITHINTRANSFORM Apply within (demeaning) transformation
yW = y;
for i = 1:n
    idx = (idIdx == i);
    if sum(idx) > 0
        yW(idx) = y(idx) - mean(y(idx));
    end
end
end

function yW = withinTransformTime(y, timeIdx, T)
%WITHINTRANSFORMTIME Apply time demeaning
yW = y;
for t = 1:T
    idx = (timeIdx == t);
    if sum(idx) > 0
        yW(idx) = y(idx) - mean(y(idx));
    end
end
end

function foldIdx = createFolds(idIdx, n, nFolds)
%CREATEFOLDS Create cross-validation fold indices
N = length(idIdx);
foldIdx = zeros(N, 1);

% Assign folds at the individual level (to avoid data leakage)
indivFold = mod(randperm(n), nFolds) + 1;

for i = 1:n
    idx = (idIdx == i);
    foldIdx(idx) = indivFold(i);
end
end

function yHat = mlPredict(XTrain, yTrain, XTest, opts)
%MLPREDICT Fit ML model and predict

switch opts.method
    case 'lasso'
        yHat = lassoPredict(XTrain, yTrain, XTest, opts.lambda);

    case 'ridge'
        yHat = ridgePredict(XTrain, yTrain, XTest, opts.lambda);

    case 'elasticnet'
        yHat = elasticNetPredict(XTrain, yTrain, XTest, opts.lambda);

    case 'rf'
        yHat = rfPredict(XTrain, yTrain, XTest);
end
end

function yHat = lassoPredict(XTrain, yTrain, XTest, lambda)
%LASSOPREDICT LASSO prediction with CV

% Standardize
[XTrainStd, mu, sigma] = zscore(XTrain);
XTestStd = (XTest - mu) ./ (sigma + 1e-10);

if isempty(lambda)
    % Cross-validation for lambda
    lambdaSeq = logspace(-4, 1, 50);
    nLambda = length(lambdaSeq);
    cv = 5;
    cvMSE = zeros(nLambda, 1);
    nTrain = size(XTrainStd, 1);

    for l = 1:nLambda
        foldMSE = zeros(cv, 1);
        cvFolds = mod(1:nTrain, cv) + 1;

        for fold = 1:cv
            cvTest = (cvFolds == fold);
            cvTrain = ~cvTest;

            beta = lassoFit(XTrainStd(cvTrain, :), yTrain(cvTrain), lambdaSeq(l));
            yPred = XTrainStd(cvTest, :) * beta;
            foldMSE(fold) = mean((yTrain(cvTest) - yPred).^2);
        end
        cvMSE(l) = mean(foldMSE);
    end

    [~, bestIdx] = min(cvMSE);
    lambda = lambdaSeq(bestIdx);
end

% Fit final model
beta = lassoFit(XTrainStd, yTrain, lambda);
yHat = XTestStd * beta;
end

function beta = lassoFit(X, y, lambda)
%LASSOFIT Coordinate descent LASSO
[n, p] = size(X);
beta = zeros(p, 1);

maxIter = 100;
tol = 1e-4;

% Precompute
XX = X' * X;
Xy = X' * y;

for iter = 1:maxIter
    betaOld = beta;

    for j = 1:p
        % Partial residual
        r = Xy(j) - XX(j, :) * beta + XX(j, j) * beta(j);

        % Soft thresholding
        beta(j) = softThreshold(r, lambda * n) / XX(j, j);
    end

    if max(abs(beta - betaOld)) < tol
        break;
    end
end
end

function z = softThreshold(x, lambda)
%SOFTTHRESHOLD Soft thresholding operator
z = sign(x) .* max(abs(x) - lambda, 0);
end

function yHat = ridgePredict(XTrain, yTrain, XTest, lambda)
%RIDGEPREDICT Ridge regression prediction

[XTrainStd, mu, sigma] = zscore(XTrain);
XTestStd = (XTest - mu) ./ (sigma + 1e-10);

[~, p] = size(XTrainStd);

if isempty(lambda)
    lambda = 1;  % Default
end

beta = (XTrainStd' * XTrainStd + lambda * eye(p)) \ (XTrainStd' * yTrain);
yHat = XTestStd * beta;
end

function yHat = elasticNetPredict(XTrain, yTrain, XTest, lambda)
%ELASTICNETPREDICT Elastic net prediction (alpha = 0.5 mix)

[XTrainStd, mu, sigma] = zscore(XTrain);
XTestStd = (XTest - mu) ./ (sigma + 1e-10);

if isempty(lambda)
    lambda = 0.1;
end

alpha = 0.5;  % Mix between LASSO and Ridge

[n, p] = size(XTrainStd);
beta = zeros(p, 1);

maxIter = 100;
tol = 1e-4;

XX = XTrainStd' * XTrainStd;
Xy = XTrainStd' * yTrain;

for iter = 1:maxIter
    betaOld = beta;

    for j = 1:p
        r = Xy(j) - XX(j, :) * beta + XX(j, j) * beta(j);

        % Elastic net update
        beta(j) = softThreshold(r, alpha * lambda * n) / ...
            (XX(j, j) + (1 - alpha) * lambda * n);
    end

    if max(abs(beta - betaOld)) < tol
        break;
    end
end

yHat = XTestStd * beta;
end

function yHat = rfPredict(XTrain, yTrain, XTest)
%RFPREDICT Random Forest prediction

% Check if TreeBagger is available
if exist('TreeBagger', 'class')
    nTrees = 100;
    rf = TreeBagger(nTrees, XTrain, yTrain, 'Method', 'regression');
    yHat = predict(rf, XTest);
else
    % Fallback to Ridge if RF not available
    warning('TreeBagger not available, using Ridge regression');
    yHat = ridgePredict(XTrain, yTrain, XTest, 1);
end
end

function varTheta = computeClusteredVarScalar(D, res, cluster, N)
%COMPUTECLUSTEREDVARSCALAR Clustered variance for scalar parameter

[~, ~, clusterIdx] = unique(cluster);
nClusters = max(clusterIdx);

% psi = D * residual (influence function)
psi = D .* res;

% Sum within clusters
clusterSum = zeros(nClusters, 1);
for g = 1:nClusters
    clusterSum(g) = sum(psi(clusterIdx == g));
end

% Variance estimate
denom = (D' * D)^2;
numer = sum(clusterSum.^2);

% Finite sample correction
correction = nClusters / (nClusters - 1) * (N - 1) / (N - 1);

varTheta = correction * numer / denom;
end
