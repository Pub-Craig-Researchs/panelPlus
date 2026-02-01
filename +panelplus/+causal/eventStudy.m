function est = eventStudy(id, time, y, X, treatTime, varargin)
%EVENTSTUDY Event study with dynamic treatment effects
%   Estimates dynamic treatment effects relative to treatment timing.
%
%   est = EVENTSTUDY(id, time, y, X, treatTime) estimates event study model
%   where treatTime is the treatment time for each observation.
%
%   est = EVENTSTUDY(id, time, y, X, treatTime, Name, Value) with options:
%
%   Options:
%   - 'leads': Number of pre-treatment periods (default: 5)
%   - 'lags': Number of post-treatment periods (default: 10)
%   - 'reference': Reference period relative to treatment (default: -1)
%   - 'binned': Bin endpoints (default: true)
%   - 'cluster': Cluster variable for SE (default: id)
%   - 'method': 'twfe' or 'sunab' (default: 'twfe')
%
%   Output:
%   - est.coef: Coefficients for each relative time period
%   - est.methodDetails.relTime: Relative time periods
%   - est.methodDetails.preTestF: F-test for pre-trends
%
%   Example:
%       est = panelplus.causal.eventStudy(id, time, y, [], treatTime, ...
%                   'leads', 5, 'lags', 10);
%       panelplus.causal.eventStudyPlot(est);
%
%   See also DID, STAGGEREDDID, EVENTSTUDYPLOT
%
%   Copyright 2026 PanelPlus Toolbox

arguments
    id (:,1)
    time (:,1)
    y (:,1) double
    X (:,:) double
    treatTime (:,1) double
end
arguments (Repeating)
    varargin
end

% Parse options
p = inputParser;
addParameter(p, 'leads', 5, @isnumeric);
addParameter(p, 'lags', 10, @isnumeric);
addParameter(p, 'reference', -1, @isnumeric);
addParameter(p, 'binned', true, @islogical);
addParameter(p, 'cluster', id, @(x) length(x) == length(y));
addParameter(p, 'method', 'twfe', @(x) ismember(x, {'twfe', 'sunab'}));
parse(p, varargin{:});
opts = p.Results;

% Initialize result
est = panelplus.utils.panelResult();
est.method = "Event Study";
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

% Compute relative time
relTime = time - treatTime;

% Handle never-treated (set to very negative)
neverTreated = isinf(treatTime);
relTime(neverTreated) = -999;

% Define relative time periods for estimation
minLead = -opts.leads;
maxLag = opts.lags;
ref = opts.reference;

% Create relative time indicators
relTimePeriods = minLead:maxLag;
relTimePeriods = relTimePeriods(relTimePeriods ~= ref);  % Exclude reference
nPeriods = length(relTimePeriods);

% Build event study dummies
EventDummies = zeros(N, nPeriods);

for j = 1:nPeriods
    rt = relTimePeriods(j);

    if opts.binned
        if rt == minLead
            % Bin all periods before minLead
            EventDummies(:, j) = double(relTime <= rt & ~neverTreated);
        elseif rt == maxLag
            % Bin all periods after maxLag
            EventDummies(:, j) = double(relTime >= rt & ~neverTreated);
        else
            EventDummies(:, j) = double(relTime == rt & ~neverTreated);
        end
    else
        EventDummies(:, j) = double(relTime == rt & ~neverTreated);
    end
end

% Apply within transformation (individual FE)
yW = withinTransform(y, idIdx, n);
EventDummiesW = zeros(size(EventDummies));
for j = 1:nPeriods
    EventDummiesW(:, j) = withinTransform(EventDummies(:, j), idIdx, n);
end

% Add controls if provided
if ~isempty(X)
    k = size(X, 2);
    XW = zeros(size(X));
    for j = 1:k
        XW(:, j) = withinTransform(X(:, j), idIdx, n);
    end
    XFull = [EventDummiesW, XW];
else
    k = 0;
    XFull = EventDummiesW;
end

% Add time fixed effects
TimeDummies = zeros(N, T-1);
for t = 2:T
    TimeDummies(:, t-1) = double(timeIdx == t);
end
TimeDummiesW = zeros(size(TimeDummies));
for j = 1:T-1
    TimeDummiesW(:, j) = withinTransform(TimeDummies(:, j), idIdx, n);
end
XFull = [XFull, TimeDummiesW];

% OLS estimation
kFull = size(XFull, 2);
coef = (XFull' * XFull) \ (XFull' * yW);
res = yW - XFull * coef;

% Variance estimation with clustering
invXX = (XFull' * XFull) \ eye(kFull);
varcoef = computeClusteredVar(XFull, res, opts.cluster, invXX, N, kFull);

% Extract event study coefficients
eventCoef = coef(1:nPeriods);
eventVarcoef = varcoef(1:nPeriods, 1:nPeriods);
eventStderr = sqrt(diag(eventVarcoef));

% Add reference period (0 by construction)
fullPeriods = minLead:maxLag;
nFullPeriods = length(fullPeriods);
fullCoef = zeros(nFullPeriods, 1);
fullStderr = zeros(nFullPeriods, 1);

for j = 1:nFullPeriods
    rt = fullPeriods(j);
    if rt == ref
        fullCoef(j) = 0;
        fullStderr(j) = 0;
    else
        idx = find(relTimePeriods == rt);
        if ~isempty(idx)
            fullCoef(j) = eventCoef(idx);
            fullStderr(j) = eventStderr(idx);
        end
    end
end

% t-statistics and p-values
tstat = fullCoef ./ (fullStderr + 1e-10);
tstat(fullPeriods == ref) = NaN;
pval = 2 * (1 - tcdf(abs(tstat), N - kFull));

% Pre-trend F-test
preIdx = find(relTimePeriods < 0);
if length(preIdx) > 1
    preTrendCoef = eventCoef(preIdx);
    preTrendVar = eventVarcoef(preIdx, preIdx);
    FstatPre = preTrendCoef' * pinv(preTrendVar) * preTrendCoef / length(preIdx);
    preDf1 = length(preIdx);
    preDf2 = N - kFull;
    prePval = 1 - fcdf(FstatPre, preDf1, preDf2);
else
    FstatPre = NaN;
    preDf1 = NaN;
    preDf2 = NaN;
    prePval = NaN;
end

% Goodness of fit
RSS = res' * res;
yMean = mean(y);
TSS = (y - yMean)' * (y - yMean);
r2 = 1 - RSS / TSS;

% Store results
est.coef = fullCoef;
est.stderr = fullStderr;
est.tstat = tstat;
est.pval = pval;
est.varcoef = eventVarcoef;
est.k = nFullPeriods;
est.r2 = r2;
est.res = res;
est.resdf = N - kFull;

% Method details
est.methodDetails.relTime = fullPeriods;
est.methodDetails.reference = ref;
est.methodDetails.binned = opts.binned;
est.methodDetails.preTestF = struct('stat', FstatPre, 'df1', preDf1, ...
    'df2', preDf2, 'pval', prePval);
est.methodDetails.controlCoef = coef(nPeriods+1:nPeriods+k);

% Variable names
est.xnames = cell(nFullPeriods, 1);
for j = 1:nFullPeriods
    rt = fullPeriods(j);
    if rt < 0
        est.xnames{j} = sprintf('t%d', rt);
    elseif rt == 0
        est.xnames{j} = 't0';
    else
        est.xnames{j} = sprintf('t+%d', rt);
    end
end

est.isLinear = true;
est.hasIndividualEffects = true;
est.hasTimeEffects = true;
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
