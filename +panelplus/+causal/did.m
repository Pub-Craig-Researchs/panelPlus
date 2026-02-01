function est = did(id, time, y, X, treat, post, varargin)
%DID Difference-in-Differences estimation
%   Computes canonical and multi-period DID with two-way fixed effects.
%
%   est = DID(id, time, y, X, treat, post) estimates:
%       y_it = alpha_i + gamma_t + delta*(treat_i * post_t) + X_it*beta + e_it
%
%   est = DID(id, time, y, X, treat, post, Name, Value) with additional options
%
%   Inputs:
%   - id, time: Panel identifiers
%   - y: Outcome variable
%   - X: Control variables (can be empty [])
%   - treat: Treatment indicator (time-invariant binary)
%   - post: Post-treatment indicator (binary)
%
%   Options:
%   - 'method': Estimation method
%       'twfe'    - Two-way fixed effects (default)
%       'simple'  - Simple 2x2 DID
%       'firstdiff' - First-difference DID
%   - 'cluster': Cluster variable for SE (default: id)
%   - 'multiway': Use multi-way clustering (default: false)
%   - 'pretrend': Run pre-trend test (default: true)
%   - 'timeInteraction': Include treat × time dummies (default: false)
%
%   Example:
%       % Basic DID
%       est = panelplus.causal.did(id, time, y, [], treat, post);
%
%       % DID with controls and id-level clustering
%       est = panelplus.causal.did(id, time, y, X, treat, post, 'cluster', id);
%
%   References:
%   - Angrist, J.D. and Pischke, J.S. (2009). Mostly Harmless Econometrics
%   - Goodman-Bacon, A. (2021). Difference-in-Differences with Variation in Treatment Timing
%
%   See also STAGGEREDDID, EVENTSTUDY
%
%   Copyright 2026 PanelPlus Toolbox

arguments
    id (:,1)
    time (:,1)
    y (:,1) double
    X (:,:) double
    treat (:,1) double
    post (:,1) double
end
arguments (Repeating)
    varargin
end

% Parse options
p = inputParser;
addParameter(p, 'method', 'twfe', @(x) ismember(x, {'twfe', 'simple', 'firstdiff'}));
addParameter(p, 'cluster', id, @(x) length(x) == length(y));
addParameter(p, 'multiway', false, @islogical);
addParameter(p, 'pretrend', true, @islogical);
addParameter(p, 'timeInteraction', false, @islogical);
parse(p, varargin{:});
opts = p.Results;

% Initialize result
est = panelplus.utils.panelResult();
est.method = "Difference-in-Differences";
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

est.n = n;
est.T = T;
est.N = N;

% Create DID interaction term
D = treat .* post;  % Treatment effect indicator

switch opts.method
    case 'twfe'
        % Two-way fixed effects estimation
        [est, yResid, XResid] = estimateTWFE(y, X, D, treat, post, ...
            idIdx, timeIdx, n, T, N, opts);

    case 'simple'
        % Simple 2x2 DID (mean comparison)
        [est] = estimateSimpleDID(y, treat, post, D, opts);

    case 'firstdiff'
        % First-difference DID
        [est] = estimateFirstDiffDID(y, X, treat, post, D, id, time, opts);
end

% Pre-trend test
if opts.pretrend
    preTrendTest = runPreTrendTest(y, X, treat, post, id, time, opts);
    est.diagnostics.tests{end+1} = preTrendTest;
end

% Parallel trends visual diagnostic data
est.methodDetails.treatMeans = computeGroupMeans(y, treat, time, utime);
est.methodDetails.controlMeans = computeGroupMeans(y, 1-treat, time, utime);

est.isLinear = true;
est.hasIndividualEffects = strcmp(opts.method, 'twfe');
est.hasTimeEffects = strcmp(opts.method, 'twfe');
end

%% Estimation Methods

function [est, yResid, XResid] = estimateTWFE(y, X, D, treat, post, idIdx, timeIdx, n, T, N, opts)
%ESTIMATETWFE Two-way fixed effects DID estimation

k = size(X, 2);

% Create fixed effects dummies
% Individual FE (leave out first for identification)
idDummies = zeros(N, n-1);
for i = 2:n
    idDummies(:, i-1) = double(idIdx == i);
end

% Time FE (leave out first)
timeDummies = zeros(N, T-1);
for t = 2:T
    timeDummies(:, t-1) = double(timeIdx == t);
end

% Build regression matrix
if isempty(X)
    XFull = [D, idDummies, timeDummies];
    varNames = [{'ATT'}; repmat({'FE_id'}, n-1, 1); repmat({'FE_t'}, T-1, 1)];
else
    XFull = [D, X, idDummies, timeDummies];
    varNames = [{'ATT'}; repmat({'control'}, k, 1); repmat({'FE_id'}, n-1, 1); repmat({'FE_t'}, T-1, 1)];
end

% OLS estimation
kFull = size(XFull, 2);
coef = (XFull' * XFull) \ (XFull' * y);

% Residuals
yhat = XFull * coef;
res = y - yhat;

% Variance estimation with clustering
invXX = (XFull' * XFull) \ eye(kFull);

if opts.multiway
    % Two-way clustering (id × time)
    varcoef = panelplus.utils.multiWayCluster(XFull, res, {opts.cluster, timeIdx});
else
    % One-way clustering
    varcoef = computeClusteredVar(XFull, res, opts.cluster, invXX, N, kFull);
end

% Standard errors for ATT (and controls if present)
nReportCoef = 1 + k;  % ATT + controls
stderr = sqrt(diag(varcoef(1:nReportCoef, 1:nReportCoef)));

% t-statistics and p-values
tstat = coef(1:nReportCoef) ./ stderr;
pval = 2 * (1 - tcdf(abs(tstat), N - kFull));

% Goodness of fit
RSS = res' * res;
yMean = mean(y);
TSS = (y - yMean)' * (y - yMean);
r2 = 1 - RSS / TSS;
adjr2 = 1 - (1 - r2) * (N - 1) / (N - kFull);

% Store results
est = panelplus.utils.panelResult();
est.method = "DID (TWFE)";
est.coef = coef(1:nReportCoef);
est.varcoef = varcoef(1:nReportCoef, 1:nReportCoef);
est.stderr = stderr;
est.tstat = tstat;
est.pval = pval;
est.yhat = yhat;
est.res = res;
est.resdf = N - kFull;
est.r2 = r2;
est.adjr2 = adjr2;
est.RSS = RSS;
est.TSS = TSS;
est.N = N;
est.k = nReportCoef;

% ATT is the first coefficient
est.methodDetails.ATT = coef(1);
est.methodDetails.ATT_se = stderr(1);
est.methodDetails.ATT_pval = pval(1);

% Individual and time effects
est.individualEffects = coef(nReportCoef+1:nReportCoef+n-1);
est.timeEffects = coef(nReportCoef+n:end);

yResid = res;
XResid = XFull;
end

function est = estimateSimpleDID(y, treat, post, ~, ~)
%ESTIMATESIMPLEDID Simple 2x2 DID estimation

% Four group means
y11 = mean(y(treat == 1 & post == 1));  % Treated, post
y10 = mean(y(treat == 1 & post == 0));  % Treated, pre
y01 = mean(y(treat == 0 & post == 1));  % Control, post
y00 = mean(y(treat == 0 & post == 0));  % Control, pre

% DID estimator
ATT = (y11 - y10) - (y01 - y00);

% Bootstrap standard error
nBoot = 1000;
N = length(y);
bootATT = zeros(nBoot, 1);

for b = 1:nBoot
    bootIdx = randsample(N, N, true);
    yb = y(bootIdx);
    treatb = treat(bootIdx);
    postb = post(bootIdx);

    y11b = mean(yb(treatb == 1 & postb == 1));
    y10b = mean(yb(treatb == 1 & postb == 0));
    y01b = mean(yb(treatb == 0 & postb == 1));
    y00b = mean(yb(treatb == 0 & postb == 0));

    bootATT(b) = (y11b - y10b) - (y01b - y00b);
end

ATT_se = std(bootATT);

est = panelplus.utils.panelResult();
est.method = "DID (Simple 2x2)";
est.coef = ATT;
est.stderr = ATT_se;
est.tstat = ATT / ATT_se;
est.pval = 2 * (1 - normcdf(abs(ATT / ATT_se)));
est.N = N;
est.k = 1;

est.methodDetails.ATT = ATT;
est.methodDetails.y_treat_post = y11;
est.methodDetails.y_treat_pre = y10;
est.methodDetails.y_control_post = y01;
est.methodDetails.y_control_pre = y00;
est.methodDetails.bootSE = ATT_se;
end

function est = estimateFirstDiffDID(y, X, treat, ~, ~, id, time, opts)
%ESTIMATEFIRSTDIFFDID First-difference DID estimation

[uid, ~, idIdx] = unique(id);
[utime, ~, timeIdx] = unique(time);
n = length(uid);
T = length(utime);
N = length(y);

% First-difference transformation
yDiff = zeros(N - n, 1);
treatDiff = zeros(N - n, 1);
XDiff = zeros(N - n, size(X, 2));
clusterDiff = zeros(N - n, 1);

row = 1;
for i = 1:n
    idx_i = find(idIdx == i);
    Ti = length(idx_i);

    if Ti > 1
        % Sort by time
        [~, sortIdx] = sort(timeIdx(idx_i));
        idx_sorted = idx_i(sortIdx);

        for t = 2:Ti
            yDiff(row) = y(idx_sorted(t)) - y(idx_sorted(t-1));
            treatDiff(row) = treat(idx_sorted(t));  % Treatment is time-invariant
            if ~isempty(X)
                XDiff(row, :) = X(idx_sorted(t), :) - X(idx_sorted(t-1), :);
            end
            clusterDiff(row) = id(idx_sorted(t));
            row = row + 1;
        end
    end
end

% Trim to actual size
yDiff = yDiff(1:row-1);
treatDiff = treatDiff(1:row-1);
XDiff = XDiff(1:row-1, :);
clusterDiff = clusterDiff(1:row-1);

% Regress Δy on treat (cross-sectional variation)
if isempty(X)
    XReg = [treatDiff, ones(length(yDiff), 1)];
else
    XReg = [treatDiff, XDiff, ones(length(yDiff), 1)];
end

k = size(XReg, 2);
Neff = length(yDiff);

coef = (XReg' * XReg) \ (XReg' * yDiff);
res = yDiff - XReg * coef;

% Clustered variance
invXX = (XReg' * XReg) \ eye(k);
varcoef = computeClusteredVar(XReg, res, clusterDiff, invXX, Neff, k);
stderr = sqrt(diag(varcoef));

est = panelplus.utils.panelResult();
est.method = "DID (First-Difference)";
est.coef = coef(1:end-1);  % Exclude constant
est.varcoef = varcoef(1:end-1, 1:end-1);
est.stderr = stderr(1:end-1);
est.tstat = coef(1:end-1) ./ stderr(1:end-1);
est.pval = 2 * (1 - tcdf(abs(est.tstat), Neff - k));
est.res = res;
est.N = Neff;
est.k = k - 1;

est.methodDetails.ATT = coef(1);
est.methodDetails.ATT_se = stderr(1);
end

%% Helper Functions

function varcoef = computeClusteredVar(X, res, cluster, invXX, N, k)
%COMPUTECLUSTEREDVAR Compute cluster-robust variance matrix
[~, ~, clusterIdx] = unique(cluster);
nClusters = max(clusterIdx);

meat = zeros(k, k);
for g = 1:nClusters
    idx = (clusterIdx == g);
    Xg = X(idx, :);
    resG = res(idx);
    sumXe = Xg' * resG;
    meat = meat + sumXe * sumXe';
end

varcoef = invXX * meat * invXX;

% Small-sample correction
correction = nClusters / (nClusters - 1) * (N - 1) / (N - k);
varcoef = correction * varcoef;
end

function preTrendTest = runPreTrendTest(y, X, treat, post, id, time, ~)
%RUNPRETRENDTEST Test for parallel pre-trends

% Subset to pre-treatment period
preIdx = (post == 0);

if sum(preIdx) < 10
    preTrendTest = struct('name', 'Pre-Trend Test', 'stat', NaN, ...
        'df', NaN, 'pval', NaN, 'warning', 'Insufficient pre-period observations');
    return;
end

yPre = y(preIdx);
treatPre = treat(preIdx);
timePre = time(preIdx);
idPre = id(preIdx);

% Test: regress y on treat × time trend in pre-period
[utime, ~, timeIdx] = unique(timePre);
T = length(utime);

% Create time trend
timeTrend = timeIdx;

% Interaction term
treatTimeTrend = treatPre .* timeTrend;

% Regression with id fixed effects
[uid, ~, idIdx] = unique(idPre);
n = length(uid);

% Within-transformation
yDemeaned = yPre;
trendDemeaned = treatTimeTrend;

for i = 1:n
    idx = (idIdx == i);
    yDemeaned(idx) = yPre(idx) - mean(yPre(idx));
    trendDemeaned(idx) = treatTimeTrend(idx) - mean(treatTimeTrend(idx));
end

% Test coefficient
beta = trendDemeaned' * yDemeaned / (trendDemeaned' * trendDemeaned);
res = yDemeaned - trendDemeaned * beta;

se = sqrt(sum(res.^2) / (length(res) - n - 1) / (trendDemeaned' * trendDemeaned));
tstat = beta / se;
pval = 2 * (1 - tcdf(abs(tstat), length(res) - n - 1));

preTrendTest = struct('name', 'Pre-Trend Test (treat × time)', ...
    'stat', tstat, 'df', length(res) - n - 1, 'pval', pval);
end

function means = computeGroupMeans(y, groupIndicator, time, utime)
%COMPUTEGROUPMEANS Compute means by group and time
T = length(utime);
means = NaN(T, 1);

for t = 1:T
    idx = (time == utime(t)) & (groupIndicator == 1);
    if sum(idx) > 0
        means(t) = mean(y(idx));
    end
end
end
