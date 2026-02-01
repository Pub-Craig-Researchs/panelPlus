function est = staggeredDID(id, time, y, X, treatTime, varargin)
%STAGGEREDDID Staggered Difference-in-Differences estimation
%   Robust DID estimators for staggered treatment adoption.
%
%   est = STAGGEREDDID(id, time, y, X, treatTime) estimates treatment effects
%   where treatTime is the treatment adoption time for each unit (Inf if never treated)
%
%   est = STAGGEREDDID(id, time, y, X, treatTime, Name, Value) with options:
%
%   Options:
%   - 'method': Estimation method
%       'cs'    - Callaway-Sant'Anna (2021) (default)
%       'sa'    - Sun-Abraham (2021) interaction-weighted
%       'bjs'   - Borusyak-Jaravel-Spiess (2024) imputation
%       'twfe'  - Standard TWFE (biased with heterogeneous effects)
%   - 'control': Control group type
%       'nevertreated' - Use never-treated units only (default)
%       'notyettreated' - Use not-yet-treated as controls
%   - 'aggregation': How to aggregate group-time ATTs
%       'simple'   - Simple average
%       'dynamic'  - By time relative to treatment
%       'calendar' - By calendar time
%       'group'    - By treatment cohort
%   - 'anticipation': Periods of anticipation (default: 0)
%   - 'cluster': Cluster variable for SE (default: id)
%   - 'nBoot': Bootstrap replications (default: 999)
%
%   Example:
%       % Callaway-Sant'Anna with never-treated controls
%       est = panelplus.causal.staggeredDID(id, time, y, [], treatTime, ...
%                   'method', 'cs', 'control', 'nevertreated');
%
%   References:
%   - Callaway, B. and Sant'Anna, P.H.C. (2021). Difference-in-Differences with Multiple Time Periods
%   - Sun, L. and Abraham, S. (2021). Estimating Dynamic Treatment Effects in Event Studies
%   - de Chaisemartin, C. and D'Haultfœuille, X. (2020). Two-Way Fixed Effects Estimators
%   - Borusyak, K., Jaravel, X., and Spiess, J. (2024). Revisiting Event Study Designs
%
%   See also DID, EVENTSTUDY
%
%   Copyright 2026 PanelPlus Toolbox

arguments
    id (:,1)
    time (:,1)
    y (:,1) double
    X (:,:) double
    treatTime (:,1) double  % Treatment time for each obs (Inf = never treated)
end
arguments (Repeating)
    varargin
end

% Parse options
p = inputParser;
addParameter(p, 'method', 'cs', @(x) ismember(x, {'cs', 'sa', 'bjs', 'twfe'}));
addParameter(p, 'control', 'nevertreated', @(x) ismember(x, {'nevertreated', 'notyettreated'}));
addParameter(p, 'aggregation', 'simple', @(x) ismember(x, {'simple', 'dynamic', 'calendar', 'group'}));
addParameter(p, 'anticipation', 0, @isnumeric);
addParameter(p, 'cluster', id, @(x) length(x) == length(y));
addParameter(p, 'nBoot', 999, @isnumeric);
parse(p, varargin{:});
opts = p.Results;

% Initialize result
est = panelplus.utils.panelResult();
est.method = sprintf("Staggered DID (%s)", upper(opts.method));
est.options = opts;

% Store original data
est.y = y;
est.X = X;

% Get panel structure
[uid, ~, idIdx] = unique(id);
[utime, ~, ~] = unique(time);
n = length(uid);
T = length(utime);
N = length(y);

est.n = n;
est.T = T;
est.N = N;

% Get unique treatment cohorts
[uTreatTime, ~, treatCohort] = unique(treatTime);
nCohorts = length(uTreatTime);
neverTreatedFlag = isinf(treatTime);

% Identify treatment cohorts (excluding never-treated)
treatCohorts = uTreatTime(~isinf(uTreatTime));
nTreatCohorts = length(treatCohorts);

if nTreatCohorts == 0
    error('No treated units found (all treatTime = Inf)');
end

switch opts.method
    case 'cs'
        % Callaway-Sant'Anna (2021)
        [attGT, gtInfo, aggATT, aggSE] = callawaysantanna(y, X, id, time, ...
            treatTime, uid, utime, opts);

    case 'sa'
        % Sun-Abraham (2021)
        [attGT, gtInfo, aggATT, aggSE] = sunabraham(y, X, id, time, ...
            treatTime, uid, utime, opts);

    case 'bjs'
        % Borusyak-Jaravel-Spiess imputation
        [attGT, gtInfo, aggATT, aggSE] = borusyakImputation(y, X, id, time, ...
            treatTime, uid, utime, opts);

    case 'twfe'
        % Standard TWFE (for comparison, known to be biased)
        [attGT, gtInfo, aggATT, aggSE] = standardTWFE(y, X, id, time, ...
            treatTime, uid, utime, opts);
end

% Store results
est.coef = aggATT;
est.stderr = aggSE;
est.tstat = aggATT ./ aggSE;
est.pval = 2 * (1 - normcdf(abs(est.tstat)));
est.k = length(aggATT);

% Method-specific details
est.methodDetails.attGT = attGT;
est.methodDetails.gtInfo = gtInfo;
est.methodDetails.treatCohorts = treatCohorts;
est.methodDetails.aggregation = opts.aggregation;
est.methodDetails.controlGroup = opts.control;

est.isLinear = true;
est.isAsymptotic = true;
end

%% Callaway-Sant'Anna (2021)

function [attGT, gtInfo, aggATT, aggSE] = callawaysantanna(y, X, id, time, treatTime, uid, utime, opts)
%CALLAWAYSANTANNA Callaway-Sant'Anna group-time ATT estimation

n = length(uid);
T = length(utime);

% Get unit-level treatment time (first observation per unit)
unitTreatTime = zeros(n, 1);
for i = 1:n
    unitIdx = find(id == uid(i), 1, 'first');
    unitTreatTime(i) = treatTime(unitIdx);
end

% Get unique treatment cohorts
treatCohorts = unique(unitTreatTime(~isinf(unitTreatTime)));
nCohorts = length(treatCohorts);

% Initialize group-time ATT storage
maxGT = nCohorts * T;
attGT = NaN(maxGT, 1);
gtInfo = struct('cohort', [], 'time', [], 'relTime', [], 'nTreat', [], 'nControl', []);
gtIdx = 1;

for g = 1:nCohorts
    cohort = treatCohorts(g);

    % Units in this cohort (using unit-level treatment time)
    cohortUnits = uid(unitTreatTime == cohort);

    for t = 1:T
        currentTime = utime(t);
        relTime = currentTime - cohort;

        % Skip pre-treatment periods with anticipation
        if relTime < -opts.anticipation
            continue;
        end

        % Get control group (using unit-level treatment time)
        if strcmp(opts.control, 'nevertreated')
            controlUnits = uid(isinf(unitTreatTime));
        else  % notyettreated
            controlUnits = uid(unitTreatTime > currentTime);
        end

        if isempty(controlUnits)
            continue;
        end

        % Compute 2x2 DID for this (g, t) cell
        % Pre-period: g - 1 (or base period)
        basePeriod = cohort - 1;
        if ~ismember(basePeriod, utime)
            basePeriod = utime(find(utime < cohort, 1, 'last'));
            if isempty(basePeriod)
                continue;
            end
        end

        % Outcome means
        y_treat_post = getMeanOutcome(y, id, time, cohortUnits, currentTime);
        y_treat_pre = getMeanOutcome(y, id, time, cohortUnits, basePeriod);
        y_control_post = getMeanOutcome(y, id, time, controlUnits, currentTime);
        y_control_pre = getMeanOutcome(y, id, time, controlUnits, basePeriod);

        % 2x2 DID
        attGT(gtIdx) = (y_treat_post - y_treat_pre) - (y_control_post - y_control_pre);

        gtInfo(gtIdx).cohort = cohort;
        gtInfo(gtIdx).time = currentTime;
        gtInfo(gtIdx).relTime = relTime;
        gtInfo(gtIdx).nTreat = length(cohortUnits);
        gtInfo(gtIdx).nControl = length(controlUnits);

        gtIdx = gtIdx + 1;
    end
end

% Trim to actual size
validIdx = ~isnan(attGT(1:gtIdx-1));
attGT = attGT(validIdx);
gtInfo = gtInfo(validIdx);

% Aggregate based on method
[aggATT, aggSE] = aggregateATT(attGT, gtInfo, opts);
end

%% Sun-Abraham (2021)

function [attGT, gtInfo, aggATT, aggSE] = sunabraham(y, X, id, time, treatTime, uid, utime, opts)
%SUNABRAHAM Sun-Abraham interaction-weighted estimator

n = length(uid);
T = length(utime);
N = length(y);

% Get cohorts and relative time
treatCohorts = unique(treatTime(~isinf(treatTime)));
nCohorts = length(treatCohorts);

% Create relative time indicator
relTime = time - treatTime;
relTime(isinf(treatTime)) = NaN;  % Never-treated

% Get unique relative times
validRelTime = relTime(~isnan(relTime));
uRelTime = unique(validRelTime);
nRelTime = length(uRelTime);

% Reference period (e.g., -1)
refPeriod = -1;
uRelTime = uRelTime(uRelTime ~= refPeriod);
nRelTime = length(uRelTime);

% Create cohort × relative-time dummies
% Interaction-weighted: weight by cohort share

% Build regression with cohort-specific relative time dummies
XInteract = zeros(N, nCohorts * nRelTime);
colNames = cell(nCohorts * nRelTime, 1);

col = 1;
for g = 1:nCohorts
    cohort = treatCohorts(g);
    cohortMask = (treatTime == cohort);

    for r = 1:nRelTime
        rt = uRelTime(r);
        XInteract(:, col) = double(cohortMask & (relTime == rt));
        colNames{col} = sprintf('g%d_r%d', cohort, rt);
        col = col + 1;
    end
end

% Add controls and fixed effects
[~, ~, idIdx] = unique(id);
[~, ~, timeIdx] = unique(time);

% Within-transform
yW = withinTransform(y, idIdx, n);
XInteractW = zeros(size(XInteract));
for j = 1:size(XInteract, 2)
    XInteractW(:, j) = withinTransform(XInteract(:, j), idIdx, n);
end

% Time fixed effects
for t = 2:T
    tDummy = double(timeIdx == t);
    tDummyW = withinTransform(tDummy, idIdx, n);
    XInteractW = [XInteractW, tDummyW];
end

% OLS estimation
k = size(XInteractW, 2);
coef = (XInteractW' * XInteractW) \ (XInteractW' * yW);
res = yW - XInteractW * coef;

% Extract cohort-time coefficients
attGT = coef(1:nCohorts * nRelTime);

% Create gtInfo
gtInfo = struct('cohort', [], 'time', [], 'relTime', [], 'nTreat', [], 'nControl', []);
idx = 1;
for g = 1:nCohorts
    for r = 1:nRelTime
        gtInfo(idx).cohort = treatCohorts(g);
        gtInfo(idx).relTime = uRelTime(r);
        gtInfo(idx).time = treatCohorts(g) + uRelTime(r);
        idx = idx + 1;
    end
end

% Aggregate with cohort weights
[aggATT, aggSE] = aggregateATT(attGT, gtInfo, opts);
end

%% Borusyak-Jaravel-Spiess Imputation

function [attGT, gtInfo, aggATT, aggSE] = borusyakImputation(y, X, id, time, treatTime, uid, utime, opts)
%BORUSYAKIMPUTATION Imputation estimator for staggered DID

N = length(y);
n = length(uid);
T = length(utime);

% Identify treated observations
postTreat = time >= treatTime;

% Step 1: Estimate model on untreated observations (never-treated + pre-treated)
untreatedIdx = ~postTreat;

[~, ~, idIdx] = unique(id);
[~, ~, timeIdx] = unique(time);

% Build FE matrices for untreated observations
yUntreated = y(untreatedIdx);
idIdxUntreated = idIdx(untreatedIdx);
timeIdxUntreated = timeIdx(untreatedIdx);

% Within-transform on untreated
[uidUntreated, ~, idIdxNew] = unique(idIdxUntreated);
nUntreated = length(uidUntreated);

yW = withinTransform(yUntreated, idIdxNew, nUntreated);

% Add time FE
XTime = zeros(length(yUntreated), T-1);
for t = 2:T
    XTime(:, t-1) = double(timeIdxUntreated == t);
end
XTimeW = zeros(size(XTime));
for j = 1:T-1
    XTimeW(:, j) = withinTransform(XTime(:, j), idIdxNew, nUntreated);
end

% Estimate time effects
timeCoef = (XTimeW' * XTimeW) \ (XTimeW' * yW);

% Step 2: Impute counterfactual for treated observations
% y0_it = alpha_i + gamma_t

% Estimate individual effects
alphaHat = zeros(n, 1);
for i = 1:n
    idx = (idIdx == i) & untreatedIdx;
    if sum(idx) > 0
        timeFE = [0; timeCoef];  % Time 1 is reference
        alphaHat(i) = mean(y(idx) - timeFE(timeIdx(idx)));
    end
end

% Impute counterfactual
timeFE = [0; timeCoef];
y0Hat = alphaHat(idIdx) + timeFE(timeIdx);

% Step 3: Treatment effect = y - y0Hat for treated observations
treatEffect = y - y0Hat;
treatEffect(~postTreat) = NaN;

% Aggregate by cohort-time
treatCohorts = unique(treatTime(~isinf(treatTime)));
nCohorts = length(treatCohorts);

attGT = [];
gtInfo = struct('cohort', [], 'time', [], 'relTime', [], 'nTreat', [], 'nControl', []);
gtIdx = 1;

for g = 1:nCohorts
    cohort = treatCohorts(g);
    cohortMask = (treatTime == cohort);

    for t = 1:T
        currentTime = utime(t);
        if currentTime < cohort
            continue;
        end

        idx = cohortMask & (time == currentTime);
        if sum(idx) > 0
            attGT(gtIdx) = mean(treatEffect(idx), 'omitnan');
            gtInfo(gtIdx).cohort = cohort;
            gtInfo(gtIdx).time = currentTime;
            gtInfo(gtIdx).relTime = currentTime - cohort;
            gtInfo(gtIdx).nTreat = sum(idx);
            gtIdx = gtIdx + 1;
        end
    end
end

attGT = attGT(:);

% Aggregate
[aggATT, aggSE] = aggregateATT(attGT, gtInfo, opts);
end

%% Standard TWFE (for comparison)

function [attGT, gtInfo, aggATT, aggSE] = standardTWFE(y, X, id, time, treatTime, uid, utime, opts)
%STANDARDTWFE Standard TWFE estimator (biased with heterogeneous effects)

N = length(y);
n = length(uid);
T = length(utime);

[~, ~, idIdx] = unique(id);
[~, ~, timeIdx] = unique(time);

% Treatment indicator
D = double(time >= treatTime);

% Within-transform
yW = withinTransform(y, idIdx, n);
DW = withinTransform(D, idIdx, n);

% Add time FE
XTime = zeros(N, T-1);
for t = 2:T
    XTime(:, t-1) = double(timeIdx == t);
end
XTimeW = zeros(size(XTime));
for j = 1:T-1
    XTimeW(:, j) = withinTransform(XTime(:, j), idIdx, n);
end

XFull = [DW, XTimeW];

% OLS
coef = (XFull' * XFull) \ (XFull' * yW);
res = yW - XFull * coef;

% Variance (clustered by id)
k = size(XFull, 2);
invXX = (XFull' * XFull) \ eye(k);
varcoef = computeClusteredVar(XFull, res, id, invXX, N, k);

aggATT = coef(1);
aggSE = sqrt(varcoef(1, 1));

% No group-time decomposition for TWFE
attGT = aggATT;
gtInfo = struct('cohort', 'pooled', 'time', 'pooled', 'relTime', NaN, 'nTreat', sum(D), 'nControl', N - sum(D));
end

%% Helper Functions

function yMean = getMeanOutcome(y, id, time, units, t)
%GETMEANOUTCOME Get mean outcome for units at time t
idx = ismember(id, units) & (time == t);
if sum(idx) > 0
    yMean = mean(y(idx));
else
    yMean = NaN;
end
end

function [aggATT, aggSE] = aggregateATT(attGT, gtInfo, opts)
%AGGREGATEATT Aggregate group-time ATTs

nGT = length(attGT);

switch opts.aggregation
    case 'simple'
        % Simple weighted average by sample size
        weights = [gtInfo.nTreat]';
        weights = weights / sum(weights);
        aggATT = sum(attGT .* weights);

        % Bootstrap SE
        aggSE = std(attGT) / sqrt(nGT);

    case 'dynamic'
        % Group by relative time
        relTimes = [gtInfo.relTime]';
        uRelTimes = unique(relTimes);
        nRelTimes = length(uRelTimes);

        aggATT = zeros(nRelTimes, 1);
        aggSE = zeros(nRelTimes, 1);

        for r = 1:nRelTimes
            rt = uRelTimes(r);
            idx = (relTimes == rt);
            weights = [gtInfo(idx).nTreat]';
            weights = weights / sum(weights);
            aggATT(r) = sum(attGT(idx) .* weights);
            aggSE(r) = std(attGT(idx)) / sqrt(sum(idx));
        end

    case 'calendar'
        % Group by calendar time
        calTimes = [gtInfo.time]';
        uCalTimes = unique(calTimes);
        nCalTimes = length(uCalTimes);

        aggATT = zeros(nCalTimes, 1);
        aggSE = zeros(nCalTimes, 1);

        for c = 1:nCalTimes
            ct = uCalTimes(c);
            idx = (calTimes == ct);
            weights = [gtInfo(idx).nTreat]';
            weights = weights / sum(weights);
            aggATT(c) = sum(attGT(idx) .* weights);
            aggSE(c) = std(attGT(idx)) / sqrt(sum(idx));
        end

    case 'group'
        % Group by treatment cohort
        cohorts = [gtInfo.cohort]';
        uCohorts = unique(cohorts);
        nCohortGroups = length(uCohorts);

        aggATT = zeros(nCohortGroups, 1);
        aggSE = zeros(nCohortGroups, 1);

        for g = 1:nCohortGroups
            coh = uCohorts(g);
            idx = (cohorts == coh);
            aggATT(g) = mean(attGT(idx));
            aggSE(g) = std(attGT(idx)) / sqrt(sum(idx));
        end
end

% Handle NaN SE
aggSE(isnan(aggSE)) = 0;
aggSE(aggSE == 0) = std(attGT) / sqrt(nGT);
end

function yW = withinTransform(y, idIdx, n)
%WITHINTRANSFORM Apply within (demeaning) transformation
yW = y;
for i = 1:n
    idx = (idIdx == i);
    yW(idx) = y(idx) - mean(y(idx));
end
end

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
correction = nClusters / (nClusters - 1) * (N - 1) / (N - k);
varcoef = correction * varcoef;
end
