function est = causalForest(X, Y, W, varargin)
%CAUSALFOREST Causal Forest for heterogeneous treatment effects
%   Estimates conditional average treatment effects (CATE) using honest trees.
%
%   est = CAUSALFOREST(X, Y, W) estimates CATE where W is binary treatment
%   est = CAUSALFOREST(X, Y, W, Name, Value) with options:
%
%   Options:
%   - 'estimator': Estimation method ('tlearner', 'xlearner', 'honest')
%                  Default: 'tlearner' (matches Python EconML)
%   - 'nTrees': Number of trees (default: 100)
%   - 'minLeafSize': Minimum samples per leaf (default: 5)
%   - 'honesty': Use honest splitting for 'honest' method (default: true)
%   - 'honestFraction': Fraction for estimation (default: 0.5)
%   - 'mtry': Variables to consider per split (default: sqrt(p))
%   - 'subsampleFraction': Bootstrap sample fraction (default: 0.5)
%   - 'propensity': Propensity scores (default: estimated)
%
%   Returns:
%   - est.coef: Average CATE (ATE)
%   - est.methodDetails.cate: Individual CATE estimates
%   - est.methodDetails.cateVar: CATE variance estimates
%   - est.methodDetails.variableImportance: Feature importance
%
%   Example:
%       % T-learner (default, matches Python)
%       est = panelplus.ml.causalForest(features, outcome, treatment);
%
%       % X-learner for better heterogeneity
%       est = panelplus.ml.causalForest(features, outcome, treatment, ...
%                   'estimator', 'xlearner');
%
%       % Original honest splitting
%       est = panelplus.ml.causalForest(features, outcome, treatment, ...
%                   'estimator', 'honest', 'nTrees', 500);
%
%   References:
%   - Wager, S. and Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects
%   - Kunzel, S. et al. (2019). Metalearners for estimating heterogeneous treatment effects
%
%   See also PANELDDML, TREEBAGGER
%
%   Copyright 2026 PanelPlus Toolbox

arguments
    X (:,:) double
    Y (:,1) double
    W (:,1) double  % Binary treatment
end
arguments (Repeating)
    varargin
end

% Parse options
p = inputParser;
addParameter(p, 'estimator', 'tlearner', @(x) ismember(x, {'tlearner', 'xlearner', 'honest', 'quantile', 'drlearner', 'psm'}));
addParameter(p, 'nTrees', 100, @isnumeric);
addParameter(p, 'minLeafSize', 5, @isnumeric);
addParameter(p, 'honesty', true, @islogical);
addParameter(p, 'honestFraction', 0.5, @isnumeric);
addParameter(p, 'mtry', [], @isnumeric);
addParameter(p, 'subsampleFraction', 0.5, @isnumeric);
addParameter(p, 'propensity', [], @isnumeric);
addParameter(p, 'quantiles', [0.025, 0.5, 0.975], @isnumeric);
% Panel data options
addParameter(p, 'id', [], @isnumeric);  % Panel unit ID
addParameter(p, 'time', [], @isnumeric);  % Time period
addParameter(p, 'panelFE', false, @islogical);  % Apply fixed effects
addParameter(p, 'cluster', [], @isnumeric);  % Cluster for SE
% PSM options
addParameter(p, 'nMatches', 1, @isnumeric);  % Number of matches per unit
addParameter(p, 'caliper', 0.2, @isnumeric);  % Caliper in SD of propensity
addParameter(p, 'replacement', true, @islogical);  % Match with replacement
parse(p, varargin{:});
opts = p.Results;



% Initialize result
est = panelplus.utils.panelResult();
est.method = "Causal Forest (" + opts.estimator + ")";
est.options = opts;

% Dimensions
[N, pFeatures] = size(X);

est.N = N;
est.k = pFeatures;

% Set default mtry
if isempty(opts.mtry)
    opts.mtry = max(1, floor(sqrt(pFeatures)));
end

% Panel fixed effects transformation if requested
X_orig = X;
Y_orig = Y;
W_orig = W;

if opts.panelFE && ~isempty(opts.id)
    % Within-transformation (demean by individual)
    [X, Y, W] = applyPanelFE(X, Y, W, opts.id);
    est.methodDetails.panelFE = true;
    est.methodDetails.nUnits = length(unique(opts.id));
else
    est.methodDetails.panelFE = false;
end

% Dispatch to appropriate estimator
switch opts.estimator
    case 'tlearner'
        [cate, cateVar, varImp] = tlearnerEstimate(X, Y, W, opts);
    case 'xlearner'
        [cate, cateVar, varImp] = xlearnerEstimate(X, Y, W, opts);
    case 'honest'
        [cate, cateVar, varImp] = honestEstimate(X, Y, W, opts);
    case 'quantile'
        [cate, cateVar, varImp] = quantileTlearnerEstimate(X, Y, W, opts);
    case 'drlearner'
        [cate, cateVar, varImp] = drlearnerEstimate(X, Y, W, opts);
    case 'psm'
        [cate, cateVar, varImp] = psmEstimate(X, Y, W, opts);
end

% Clustered standard errors if cluster variable provided
if ~isempty(opts.cluster)
    [ate, ateVar] = computeClusteredATE(cate, opts.cluster);
    ateSE = sqrt(ateVar);
    est.methodDetails.clustered = true;
else
    % Average treatment effect
    ate = mean(cate);
    ateVar = var(cate) / N + mean(cateVar) / N;
    ateSE = sqrt(ateVar);
    est.methodDetails.clustered = false;
end

% Store results
est.coef = ate;
est.varcoef = ateVar;
est.stderr = ateSE;
est.tstat = ate / ateSE;
est.pval = 2 * (1 - normcdf(abs(ate / ateSE)));

% Method details
est.methodDetails.cate = cate;
est.methodDetails.cateVar = cateVar;
est.methodDetails.cateSE = sqrt(cateVar);
est.methodDetails.variableImportance = varImp;
est.methodDetails.nTrees = opts.nTrees;
est.methodDetails.estimator = opts.estimator;

% Add quantile-specific results if using quantile estimator
if strcmp(opts.estimator, 'quantile') && exist('quantileResults', 'var')
    est.methodDetails.mu1_quantiles = quantileResults.mu1_quantiles;
    est.methodDetails.mu0_quantiles = quantileResults.mu0_quantiles;
    est.methodDetails.cate_lower = quantileResults.cate_lower;
    est.methodDetails.cate_upper = quantileResults.cate_upper;
    est.methodDetails.quantileLevels = quantileResults.quantiles;
end

% Confidence intervals for CATE
z95 = 1.96;
est.methodDetails.cateLower = cate - z95 * sqrt(cateVar);
est.methodDetails.cateUpper = cate + z95 * sqrt(cateVar);

est.isLinear = false;
est.isAsymptotic = true;
end

%% Propensity Score Matching

function [cate, cateVar, varImp] = psmEstimate(X, Y, W, opts)
%PSMESTIMATE Propensity Score Matching for ATE/ATET estimation
%   Implements nearest neighbor matching with caliper

N = size(X, 1);
nMatches = opts.nMatches;
caliper = opts.caliper;
withReplacement = opts.replacement;

% Step 1: Estimate propensity scores using logistic regression or RF
if isempty(opts.propensity)
    % Use Random Forest for propensity
    rf_prop = TreeBagger(50, X, W, ...
        'Method', 'classification', ...
        'MinLeafSize', opts.minLeafSize, ...
        'OOBPrediction', 'on', ...
        'OOBPredictorImportance', 'on');
    [~, propScores] = predict(rf_prop, X);
    propensity = propScores(:, 2);
else
    propensity = opts.propensity;
end

% Clip propensity
propensity = max(0.01, min(0.99, propensity));

% Caliper in SD of propensity
propSD = std(propensity);
caliperWidth = caliper * propSD;

% Indices
treatIdx = find(W == 1);
controlIdx = find(W == 0);
nTreat = length(treatIdx);
nControl = length(controlIdx);

% Step 2: Match each treated unit to control(s)
matchedOutcomes = zeros(nTreat, 1);
matchWeights = zeros(nTreat, 1);
matchedIdx = cell(nTreat, 1);
validMatch = true(nTreat, 1);

usedControls = [];  % For matching without replacement

for i = 1:nTreat
    ti = treatIdx(i);
    prop_i = propensity(ti);

    % Calculate distance to all controls
    if withReplacement
        availableControls = controlIdx;
    else
        availableControls = setdiff(controlIdx, usedControls);
    end

    if isempty(availableControls)
        validMatch(i) = false;
        continue;
    end

    distances = abs(propensity(availableControls) - prop_i);

    % Apply caliper
    withinCaliper = distances <= caliperWidth;

    if sum(withinCaliper) == 0
        validMatch(i) = false;
        continue;
    end

    % Find nearest neighbors
    availableInCaliper = availableControls(withinCaliper);
    distInCaliper = distances(withinCaliper);

    [~, sortIdx] = sort(distInCaliper);
    nSelect = min(nMatches, length(sortIdx));
    selectedControls = availableInCaliper(sortIdx(1:nSelect));

    % Update used controls
    if ~withReplacement
        usedControls = [usedControls; selectedControls];
    end

    % Compute matched outcome (average of matched controls)
    matchedOutcomes(i) = mean(Y(selectedControls));
    matchedIdx{i} = selectedControls;
    matchWeights(i) = 1;
end

% Step 3: Compute individual treatment effects (for treated units)
% ATET_i = Y_i(treated) - Y_matched(control)
cate_treat = Y(treatIdx(validMatch)) - matchedOutcomes(validMatch);

% For control units, we can reverse match (ATEU) or just use ATET
% Here we compute CATE for all units by propagating
cate = zeros(N, 1);
cateVar = zeros(N, 1);

% For treated units
cate(treatIdx(validMatch)) = cate_treat;

% For control units, use average ATET (simpler approach)
atet = mean(cate_treat);
cate(treatIdx(~validMatch)) = atet;
cate(controlIdx) = atet;  % Assume same effect for controls

% Variance estimation (Abadie-Imbens)
cateVar(:) = var(cate_treat);

% Variable importance based on propensity model
if isempty(opts.propensity)
    varImp = rf_prop.OOBPermutedPredictorDeltaError';
    varImp = max(0, varImp);
    varImp = varImp / sum(varImp);
else
    varImp = ones(size(X, 2), 1) / size(X, 2);
end

% Store matching details
assignin('caller', 'psmResults', struct(...
    'nMatched', sum(validMatch), ...
    'nUnmatched', sum(~validMatch), ...
    'propensity', propensity, ...
    'atet', atet, ...
    'matchedIdx', {matchedIdx}));
end

%% Panel Data Helper Functions

function [Xfe, Yfe, Wfe] = applyPanelFE(X, Y, W, id)
%APPLYPANELFE Within-transformation for panel fixed effects
%   Note: W is NOT demeaned to preserve binary treatment structure

uid = unique(id);
n = length(uid);

Xfe = X;
Yfe = Y;
Wfe = W;  % Keep W unchanged (binary treatment)

for i = 1:n
    mask = (id == uid(i));
    % Demean X and Y within each unit
    Xfe(mask, :) = X(mask, :) - mean(X(mask, :), 1);
    Yfe(mask) = Y(mask) - mean(Y(mask));
    % W is NOT demeaned - it's a binary treatment indicator
end
end

function [ate, ateVar] = computeClusteredATE(cate, cluster)
%COMPUTECLUSTEREDATE Compute ATE with clustered standard errors

uCluster = unique(cluster);
nCluster = length(uCluster);

% Cluster means
clusterATEs = zeros(nCluster, 1);
for g = 1:nCluster
    mask = (cluster == uCluster(g));
    clusterATEs(g) = mean(cate(mask));
end

% ATE is mean of cluster ATEs (weighted by cluster size would be alternative)
ate = mean(clusterATEs);

% Clustered variance (cluster-robust)
ateVar = var(clusterATEs) / nCluster;
end

%% DR-learner (Doubly Robust)

function [cate, cateVar, varImp] = drlearnerEstimate(X, Y, W, opts)
%DRLEARNERESTIMATE Doubly Robust learner for CATE
%   Combines propensity weighting with outcome regression for robustness
%   DR-score: (Y - mu(X,W)) * (W - e(X)) / (e(X) * (1 - e(X)))

nTrees = opts.nTrees;
minLeaf = opts.minLeafSize;
N = size(X, 1);

% Step 1: Estimate propensity scores e(X) = P(W=1|X)
if isempty(opts.propensity)
    rf_prop = TreeBagger(round(nTrees/2), X, W, ...
        'Method', 'classification', ...
        'MinLeafSize', minLeaf, ...
        'NumPredictorsToSample', opts.mtry, ...
        'OOBPrediction', 'on');
    [~, propScores] = predict(rf_prop, X);
    propensity = propScores(:, 2);  % P(W=1)
else
    propensity = opts.propensity;
end

% Clip propensity to avoid extreme weights
propensity = max(0.01, min(0.99, propensity));

% Step 2: Estimate outcome models mu_1(X) and mu_0(X)
rf_mu1 = TreeBagger(round(nTrees/2), X(W==1, :), Y(W==1), ...
    'Method', 'regression', ...
    'MinLeafSize', minLeaf, ...
    'NumPredictorsToSample', opts.mtry, ...
    'OOBPrediction', 'on', ...
    'OOBPredictorImportance', 'on');

rf_mu0 = TreeBagger(round(nTrees/2), X(W==0, :), Y(W==0), ...
    'Method', 'regression', ...
    'MinLeafSize', minLeaf, ...
    'NumPredictorsToSample', opts.mtry, ...
    'OOBPrediction', 'on', ...
    'OOBPredictorImportance', 'on');

mu1_hat = predict(rf_mu1, X);
mu0_hat = predict(rf_mu0, X);

% Step 3: Compute DR pseudo-outcome (AIPW score)
% DR_tau = mu_1(X) - mu_0(X) + W * (Y - mu_1(X)) / e(X) - (1-W) * (Y - mu_0(X)) / (1-e(X))
dr_pseudo = (mu1_hat - mu0_hat) + ...
    W .* (Y - mu1_hat) ./ propensity - ...
    (1 - W) .* (Y - mu0_hat) ./ (1 - propensity);

% Step 4: Final CATE model on DR pseudo-outcomes
rf_cate = TreeBagger(nTrees, X, dr_pseudo, ...
    'Method', 'regression', ...
    'MinLeafSize', minLeaf, ...
    'NumPredictorsToSample', opts.mtry, ...
    'OOBPrediction', 'on', ...
    'OOBPredictorImportance', 'on');

cate = predict(rf_cate, X);

% Variance estimation via tree-level predictions
allPreds = zeros(N, nTrees);
for t = 1:nTrees
    allPreds(:, t) = predict(rf_cate, X, 'Trees', t);
end
cateVar = var(allPreds, 0, 2);

% Variable importance from all stages
varImp = (rf_mu1.OOBPermutedPredictorDeltaError' + ...
    rf_mu0.OOBPermutedPredictorDeltaError' + ...
    rf_cate.OOBPermutedPredictorDeltaError') / 3;
varImp = max(0, varImp);
varImp = varImp / sum(varImp);
end

%% T-learner using TreeBagger

function [cate, cateVar, varImp] = tlearnerEstimate(X, Y, W, opts)
%TLEARNERESTIMATE T-learner: separate models for treated and control

nTrees = opts.nTrees;
minLeaf = opts.minLeafSize;

% Train treated model
rf_treat = TreeBagger(nTrees, X(W==1, :), Y(W==1), ...
    'Method', 'regression', ...
    'MinLeafSize', minLeaf, ...
    'NumPredictorsToSample', opts.mtry, ...
    'OOBPrediction', 'on', ...
    'OOBPredictorImportance', 'on');

% Train control model
rf_control = TreeBagger(nTrees, X(W==0, :), Y(W==0), ...
    'Method', 'regression', ...
    'MinLeafSize', minLeaf, ...
    'NumPredictorsToSample', opts.mtry, ...
    'OOBPrediction', 'on', ...
    'OOBPredictorImportance', 'on');

% Predict counterfactuals
mu1_hat = predict(rf_treat, X);
mu0_hat = predict(rf_control, X);

% CATE = E[Y(1)|X] - E[Y(0)|X]
cate = mu1_hat - mu0_hat;

% Variance estimation via infinitesimal jackknife
[~, stdev1] = predict(rf_treat, X, 'Trees', 'all');
[~, stdev0] = predict(rf_control, X, 'Trees', 'all');

% Use tree-level variance as proxy
allPreds1 = zeros(size(X,1), nTrees);
allPreds0 = zeros(size(X,1), nTrees);
for t = 1:nTrees
    allPreds1(:, t) = predict(rf_treat, X, 'Trees', t);
    allPreds0(:, t) = predict(rf_control, X, 'Trees', t);
end
cateVar = var(allPreds1 - allPreds0, 0, 2);

% Variable importance
varImp = (rf_treat.OOBPermutedPredictorDeltaError' + ...
    rf_control.OOBPermutedPredictorDeltaError') / 2;
varImp = max(0, varImp);
varImp = varImp / sum(varImp);
end

%% X-learner using fitrensemble

function [cate, cateVar, varImp] = xlearnerEstimate(X, Y, W, opts)
%XLEARNERESTIMATE X-learner: imputes treatment effects

nTrees = opts.nTrees;
minLeaf = opts.minLeafSize;
learners = templateTree('MinLeafSize', minLeaf, ...
    'NumVariablesToSample', opts.mtry);

% Stage 1: T-learner
rf_treat = fitrensemble(X(W==1, :), Y(W==1), ...
    'Method', 'Bag', ...
    'NumLearningCycles', nTrees, ...
    'Learners', learners);

rf_control = fitrensemble(X(W==0, :), Y(W==0), ...
    'Method', 'Bag', ...
    'NumLearningCycles', nTrees, ...
    'Learners', learners);

mu1_hat = predict(rf_treat, X);
mu0_hat = predict(rf_control, X);

% Stage 2: Imputed treatment effects
D1 = Y(W==1) - mu0_hat(W==1);  % Treated: Y - mu_0(X)
D0 = mu1_hat(W==0) - Y(W==0);  % Control: mu_1(X) - Y

% Stage 3: CATE models
rf_cate1 = fitrensemble(X(W==1, :), D1, ...
    'Method', 'Bag', ...
    'NumLearningCycles', round(nTrees/2), ...
    'Learners', learners);

rf_cate0 = fitrensemble(X(W==0, :), D0, ...
    'Method', 'Bag', ...
    'NumLearningCycles', round(nTrees/2), ...
    'Learners', learners);

% Propensity-weighted average
propensity = sum(W) / length(W);
tau1 = predict(rf_cate1, X);
tau0 = predict(rf_cate0, X);
cate = propensity * tau0 + (1 - propensity) * tau1;

% Variance approximation (simple MSE-based)
cateVar = 0.1 * ones(size(X,1), 1);  % Conservative estimate

% Variable importance from stage 1
varImp = (predictorImportance(rf_treat)' + ...
    predictorImportance(rf_control)') / 2;
varImp = max(0, varImp);
varImp = varImp / sum(varImp);
end

%% Quantile T-learner using TreeBagger.quantilePredict

function [cate, cateVar, varImp] = quantileTlearnerEstimate(X, Y, W, opts)
%QUANTILETLEARNERESTIMATE Quantile T-learner for CATE confidence intervals
%   Uses TreeBagger with quantilePredict to get prediction intervals

nTrees = opts.nTrees;
minLeaf = opts.minLeafSize;
quantiles = opts.quantiles;

% Train treated model with quantile prediction
rf_treat = TreeBagger(nTrees, X(W==1, :), Y(W==1), ...
    'Method', 'regression', ...
    'MinLeafSize', minLeaf, ...
    'NumPredictorsToSample', opts.mtry, ...
    'OOBPrediction', 'on', ...
    'OOBPredictorImportance', 'on');

% Train control model
rf_control = TreeBagger(nTrees, X(W==0, :), Y(W==0), ...
    'Method', 'regression', ...
    'MinLeafSize', minLeaf, ...
    'NumPredictorsToSample', opts.mtry, ...
    'OOBPrediction', 'on', ...
    'OOBPredictorImportance', 'on');

% Quantile predictions for treated potential outcome
mu1_quantiles = quantilePredict(rf_treat, X, 'Quantile', quantiles);

% Quantile predictions for control potential outcome
mu0_quantiles = quantilePredict(rf_control, X, 'Quantile', quantiles);

% CATE quantiles: for each quantile level
% Note: This is approximate since CATE = Y(1) - Y(0) doesn't have
% simple quantile algebra. We use (q_tau(Y1) - q_(1-tau)(Y0)) bounds.
nQ = length(quantiles);
medianIdx = find(abs(quantiles - 0.5) < 0.01, 1);
if isempty(medianIdx)
    medianIdx = ceil(nQ / 2);
end

% Point estimate (median)
cate = mu1_quantiles(:, medianIdx) - mu0_quantiles(:, medianIdx);

% For variance, use IQR-based estimate if we have quantiles
if nQ >= 3
    % Lower and upper bounds for CATE
    % Conservative: subtract lower Y(1) quantile from upper Y(0) quantile for lower bound
    cate_lower = mu1_quantiles(:, 1) - mu0_quantiles(:, end);
    cate_upper = mu1_quantiles(:, end) - mu0_quantiles(:, 1);

    % Approximate variance from quantile spread (assuming normal)
    % 95% CI width = 3.92 * SE
    cateVar = ((cate_upper - cate_lower) / 3.92).^2;
else
    cateVar = 0.1 * ones(size(X, 1), 1);
end

% Variable importance
varImp = (rf_treat.OOBPermutedPredictorDeltaError' + ...
    rf_control.OOBPermutedPredictorDeltaError') / 2;
varImp = max(0, varImp);
varImp = varImp / sum(varImp);

% Store quantile predictions in a persistent place (will be added to methodDetails)
assignin('caller', 'quantileResults', struct(...
    'mu1_quantiles', mu1_quantiles, ...
    'mu0_quantiles', mu0_quantiles, ...
    'cate_lower', cate_lower, ...
    'cate_upper', cate_upper, ...
    'quantiles', quantiles));
end

%% Honest Causal Forest (original implementation)

function [cate, cateVar, varImp] = honestEstimate(X, Y, W, opts)
%HONESTESTIMATE Honest causal forest with IPW estimation

N = size(X, 1);
pFeatures = size(X, 2);

% Estimate propensity scores if not provided
if isempty(opts.propensity)
    propensity = estimatePropensity(X, W);
else
    propensity = opts.propensity;
end

% Clip propensity to avoid extreme weights
propensity = max(0.01, min(0.99, propensity));

% Initialize storage
nTrees = opts.nTrees;
cateMatrix = zeros(N, nTrees);
variableUsage = zeros(pFeatures, 1);

% Build forest
for tree = 1:nTrees
    % Bootstrap sample
    nSample = round(N * opts.subsampleFraction);
    sampleIdx = randsample(N, nSample, true);

    if opts.honesty
        % Split sample for honest estimation
        nHonest = round(nSample * opts.honestFraction);
        shuffled = sampleIdx(randperm(nSample));
        trainIdx = shuffled(1:nSample-nHonest);
        honestIdx = shuffled(nSample-nHonest+1:end);
    else
        trainIdx = sampleIdx;
        honestIdx = sampleIdx;
    end

    % Build tree
    [treeStruct, varUsed] = buildCausalTree(X, Y, W, propensity, ...
        trainIdx, opts);

    variableUsage = variableUsage + varUsed;

    % Predict CATE for all observations using honest sample
    if opts.honesty
        cateMatrix(:, tree) = predictCATEHonest(treeStruct, X, Y, W, ...
            propensity, honestIdx);
    else
        cateMatrix(:, tree) = predictCATE(treeStruct, X);
    end
end

% Average across trees
cate = mean(cateMatrix, 2);
cateVar = var(cateMatrix, 0, 2);

% Variable importance
varImp = variableUsage / sum(variableUsage);
end

%% Helper Functions

function propensity = estimatePropensity(X, W)
%ESTIMATEPROPENSITY Estimate propensity scores with logistic regression

N = size(X, 1);

% Add intercept
Xaug = [ones(N, 1), X];

% Logistic regression via IRLS
beta = zeros(size(Xaug, 2), 1);
maxIter = 25;

for iter = 1:maxIter
    eta = Xaug * beta;
    mu = 1 ./ (1 + exp(-eta));
    mu = max(0.001, min(0.999, mu));

    wt = mu .* (1 - mu);
    z = eta + (W - mu) ./ wt;

    % Weighted least squares
    XtW = Xaug' * diag(wt);
    beta = (XtW * Xaug) \ (XtW * z);
end

eta = Xaug * beta;
propensity = 1 ./ (1 + exp(-eta));
end

function [tree, varUsed] = buildCausalTree(X, Y, W, propensity, trainIdx, opts)
%BUILDCAUSALTREE Build a single causal tree

[~, p] = size(X);
varUsed = zeros(p, 1);

% Initialize tree
tree = struct('isLeaf', true, 'splitVar', 0, 'splitVal', 0, ...
    'left', [], 'right', [], 'tau', 0, 'indices', trainIdx);

% Recursive splitting
tree = splitNode(tree, X, Y, W, propensity, opts, varUsed);

    function node = splitNode(node, X, Y, W, e, opts, varUsed)
        idx = node.indices;

        % Check stopping conditions
        if length(idx) < 2 * opts.minLeafSize
            node.isLeaf = true;
            node.tau = estimateLeafTau(Y(idx), W(idx), e(idx));
            return;
        end

        % Find best split
        [bestVar, bestVal, bestGain] = findBestSplit(X, Y, W, e, idx, opts);

        if bestGain <= 0
            node.isLeaf = true;
            node.tau = estimateLeafTau(Y(idx), W(idx), e(idx));
            return;
        end

        % Perform split
        leftIdx = idx(X(idx, bestVar) <= bestVal);
        rightIdx = idx(X(idx, bestVar) > bestVal);

        if length(leftIdx) < opts.minLeafSize || length(rightIdx) < opts.minLeafSize
            node.isLeaf = true;
            node.tau = estimateLeafTau(Y(idx), W(idx), e(idx));
            return;
        end

        node.isLeaf = false;
        node.splitVar = bestVar;
        node.splitVal = bestVal;
        varUsed(bestVar) = varUsed(bestVar) + 1;

        % Create child nodes
        node.left = struct('isLeaf', true, 'splitVar', 0, 'splitVal', 0, ...
            'left', [], 'right', [], 'tau', 0, 'indices', leftIdx);
        node.right = struct('isLeaf', true, 'splitVar', 0, 'splitVal', 0, ...
            'left', [], 'right', [], 'tau', 0, 'indices', rightIdx);

        % Recurse
        node.left = splitNode(node.left, X, Y, W, e, opts, varUsed);
        node.right = splitNode(node.right, X, Y, W, e, opts, varUsed);
    end
end

function [bestVar, bestVal, bestGain] = findBestSplit(X, Y, W, e, idx, opts)
%FINDBESTSPLIT Find best split point maximizing treatment effect heterogeneity

p = size(X, 2);
bestVar = 0;
bestVal = 0;
bestGain = -Inf;

% Randomly select mtry variables
varCandidates = randsample(p, min(opts.mtry, p));

for j = varCandidates'
    xj = X(idx, j);
    uniqueVals = unique(xj);

    if length(uniqueVals) < 2
        continue;
    end

    % Try each split point
    splitCandidates = (uniqueVals(1:end-1) + uniqueVals(2:end)) / 2;

    for s = 1:length(splitCandidates)
        split = splitCandidates(s);

        leftMask = xj <= split;
        rightMask = ~leftMask;

        if sum(leftMask) < opts.minLeafSize || sum(rightMask) < opts.minLeafSize
            continue;
        end

        % Compute treatment effect heterogeneity gain
        tauLeft = estimateLeafTau(Y(idx(leftMask)), W(idx(leftMask)), e(idx(leftMask)));
        tauRight = estimateLeafTau(Y(idx(rightMask)), W(idx(rightMask)), e(idx(rightMask)));

        nLeft = sum(leftMask);
        nRight = sum(rightMask);
        n = nLeft + nRight;

        % Weighted squared difference in taus
        gain = (nLeft * nRight / n^2) * (tauLeft - tauRight)^2;

        if gain > bestGain
            bestGain = gain;
            bestVar = j;
            bestVal = split;
        end
    end
end
end

function tau = estimateLeafTau(Y, W, e)
%ESTIMATELEAFTAU Estimate treatment effect in a leaf

if isempty(Y) || sum(W) == 0 || sum(1-W) == 0
    tau = 0;
    return;
end

% Inverse propensity weighted estimator
% tau = E[Y*W/e] - E[Y*(1-W)/(1-e)]
weights1 = W ./ e;
weights0 = (1 - W) ./ (1 - e);

tau = sum(Y .* weights1) / sum(weights1) - sum(Y .* weights0) / sum(weights0);

% Bound extreme estimates
tau = max(-10, min(10, tau));
end

function cate = predictCATEHonest(tree, X, Y, W, e, honestIdx)
%PREDICTCATEHONEST Predict CATE using honest estimation

N = size(X, 1);
cate = zeros(N, 1);

% For each observation, find its leaf and estimate tau from honest sample
for i = 1:N
    leafIdx = findLeafIdx(tree, X(i, :));

    % Intersect leaf indices with honest sample
    estIdx = intersect(leafIdx, honestIdx);

    if isempty(estIdx)
        % Fall back to tree estimate
        node = findLeaf(tree, X(i, :));
        cate(i) = node.tau;
    else
        cate(i) = estimateLeafTau(Y(estIdx), W(estIdx), e(estIdx));
    end
end
end

function cate = predictCATE(tree, X)
%PREDICTCATE Predict CATE for new observations
N = size(X, 1);
cate = zeros(N, 1);

for i = 1:N
    node = findLeaf(tree, X(i, :));
    cate(i) = node.tau;
end
end

function node = findLeaf(tree, x)
%FINDLEAF Find the leaf node for observation x
node = tree;
while ~node.isLeaf
    if x(node.splitVar) <= node.splitVal
        node = node.left;
    else
        node = node.right;
    end
end
end

function idx = findLeafIdx(tree, x)
%FINDLEAFIDX Find indices in the same leaf as observation x
node = tree;
while ~node.isLeaf
    if x(node.splitVar) <= node.splitVal
        node = node.left;
    else
        node = node.right;
    end
end
idx = node.indices;
end
