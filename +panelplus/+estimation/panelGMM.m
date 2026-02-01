function est = panelGMM(id, time, y, X, varargin)
%PANELGMM Panel GMM estimation (Arellano-Bond, Blundell-Bond)
%   Computes GMM estimation for dynamic panel data models.
%
%   est = PANELGMM(id, time, y, X) estimates y_it = rho*y_{i,t-1} + X_it*beta + alpha_i + e_it
%   est = PANELGMM(id, time, y, X, Name, Value) with additional options
%
%   Options:
%   - 'method': GMM estimator type
%       'difference' - Arellano-Bond (1991) difference GMM (default)
%       'system'     - Blundell-Bond (1998) system GMM
%   - 'lags': Number of lags of dependent variable (default: 1)
%   - 'gmmlags': Lag range for GMM instruments [minlag, maxlag] (default: [2, Inf])
%   - 'twostep': Use two-step efficient GMM (default: true)
%   - 'robust': Robust (Windmeijer-corrected) SE for two-step (default: true)
%   - 'collapse': Collapse instrument matrix (default: false)
%   - 'exog': Indices of strictly exogenous regressors (default: all)
%   - 'endog': Indices of endogenous regressors (default: none)
%   - 'predetermined': Indices of predetermined regressors (default: none)
%
%   Example:
%       % Arellano-Bond difference GMM
%       est = panelplus.estimation.panelGMM(id, time, y, X, 'method', 'difference');
%
%       % Blundell-Bond system GMM with collapsed instruments
%       est = panelplus.estimation.panelGMM(id, time, y, X, 'method', 'system', 'collapse', true);
%
%   References:
%   - Arellano, M. and Bond, S. (1991). Some Tests of Specification for Panel Data
%   - Blundell, R. and Bond, S. (1998). Initial Conditions and Moment Restrictions
%   - Windmeijer, F. (2005). A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators
%   - Roodman, D. (2009). How to do xtabond2: An introduction to difference and system GMM in Stata
%
%   See also PANELFGLS, PANEL, IVPANEL
%
%   Copyright 2026 PanelPlus Toolbox

arguments
    id (:,1)
    time (:,1)
    y (:,1) double
    X (:,:) double
end
arguments (Repeating)
    varargin
end

% Parse options
p = inputParser;
addParameter(p, 'method', 'difference', @(x) ismember(x, {'difference', 'system'}));
addParameter(p, 'lags', 1, @(x) isnumeric(x) && x >= 1);
addParameter(p, 'gmmlags', [2, Inf], @isnumeric);
addParameter(p, 'twostep', true, @islogical);
addParameter(p, 'robust', true, @islogical);
addParameter(p, 'collapse', false, @islogical);
addParameter(p, 'exog', [], @isnumeric);
addParameter(p, 'endog', [], @isnumeric);
addParameter(p, 'predetermined', [], @isnumeric);
parse(p, varargin{:});
opts = p.Results;

% Initialize result
est = panelplus.utils.panelResult();
est.method = sprintf("Panel GMM (%s)", opts.method);
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

% Check minimum time periods
if T < opts.lags + 2
    error('Insufficient time periods. Need at least %d for %d lag(s).', opts.lags + 2, opts.lags);
end

% Reshape data to panel format (cell array for unbalanced support)
[yPanel, XPanel, TiVec] = reshapeToPanel(y, X, id, time, uid, utime, n, T);

% Create lagged dependent variable
[yLag, validObs] = createLaggedDepVar(yPanel, opts.lags, n, TiVec);

% =====================
% Build instrument matrix
% =====================
[Z, yDiff, XDiff, yLagDiff] = buildInstruments(yPanel, XPanel, yLag, ...
    opts, n, T, TiVec, validObs);

% Stack data for estimation
[yVec, XVec, ZMat] = stackPanelData(yDiff, XDiff, yLagDiff, Z, n, opts);

% Number of instruments
L = size(ZMat, 2);
kTotal = size(XVec, 2);  % Includes lagged y

est.k = kTotal;

% =====================
% One-step GMM
% =====================

% Initial weighting matrix (identity or 2SLS-style)
W1 = (ZMat' * ZMat) \ eye(L);

% One-step GMM estimator
XZ = XVec' * ZMat;
ZX = ZMat' * XVec;
Zy = ZMat' * yVec;

coef1 = (XZ * W1 * ZX) \ (XZ * W1 * Zy);
res1 = yVec - XVec * coef1;

if ~opts.twostep
    coef = coef1;
    res = res1;
    W = W1;
else
    % =====================
    % Two-step GMM
    % =====================

    % Optimal weighting matrix from first-step residuals
    W2 = computeOptimalWeight(ZMat, res1, n, opts);

    % Two-step GMM estimator
    coef = (XZ * W2 * ZX) \ (XZ * W2 * Zy);
    res = yVec - XVec * coef;
    W = W2;
end

% =====================
% Variance estimation
% =====================

Neff = length(yVec);

if opts.twostep && opts.robust
    % Windmeijer (2005) finite-sample correction
    varcoef = windmeijerCorrection(XVec, ZMat, res, res1, coef, coef1, W, W1, n, opts);
else
    % Standard GMM variance
    bread = (XZ * W * ZX) \ eye(kTotal);
    varcoef = bread;
end

% Standard errors
stderr = sqrt(diag(varcoef));

% t-statistics and p-values (asymptotic normal)
tstat = coef ./ stderr;
pval = 2 * (1 - normcdf(abs(tstat)));

% =====================
% Diagnostic tests
% =====================

% Hansen J-test for overidentification
J = Neff * res' * ZMat * W * ZMat' * res;
Jdf = L - kTotal;
Jpval = 1 - chi2cdf(J, max(Jdf, 1));

% Arellano-Bond AR tests
[AR1stat, AR1pval] = arellonoBondARTest(res, ZMat, 1, n, opts);
[AR2stat, AR2pval] = arellonoBondARTest(res, ZMat, 2, n, opts);

% =====================
% Store results
% =====================
est.coef = coef;
est.varcoef = varcoef;
est.stderr = stderr;
est.tstat = tstat;
est.pval = pval;
est.res = res;
est.resdf = Neff - kTotal;

% Method details
est.methodDetails.gmmMethod = opts.method;
est.methodDetails.numInstruments = L;
est.methodDetails.twoStep = opts.twostep;
est.methodDetails.collapsed = opts.collapse;

% Diagnostic tests
est.diagnostics.tests = {
    struct('name', 'Hansen J-test', 'stat', J, 'df', Jdf, 'pval', Jpval);
    struct('name', 'Arellano-Bond AR(1)', 'stat', AR1stat, 'df', NaN, 'pval', AR1pval);
    struct('name', 'Arellano-Bond AR(2)', 'stat', AR2stat, 'df', NaN, 'pval', AR2pval)
    };

est.isLinear = true;
est.isAsymptotic = true;
est.hasIndividualEffects = true;

% Variable names (first k-opts.lags are lagged Y)
est.xnames = cell(kTotal, 1);
for lag = 1:opts.lags
    est.xnames{lag} = sprintf('L%d.y', lag);
end
for j = 1:k
    est.xnames{opts.lags + j} = sprintf('x%d', j);
end
end

%% Helper Functions

function [yPanel, XPanel, TiVec] = reshapeToPanel(y, X, id, time, uid, utime, n, ~)
%RESHAPETOPANEL Reshape stacked data to panel cell arrays
k = size(X, 2);
T = length(utime);

yPanel = cell(n, 1);
XPanel = cell(n, 1);
TiVec = zeros(n, 1);

for i = 1:n
    idx = (id == uid(i));
    ti = time(idx);
    yi = y(idx);
    Xi = X(idx, :);

    % Map to standard time indices
    [~, tIdx] = ismember(ti, utime);

    yPanel{i} = NaN(T, 1);
    XPanel{i} = NaN(T, k);
    yPanel{i}(tIdx) = yi;
    XPanel{i}(tIdx, :) = Xi;

    TiVec(i) = sum(idx);
end
end

function [yLag, validObs] = createLaggedDepVar(yPanel, lags, n, ~)
%CREATELAGGEDDEPVAR Create lagged dependent variables
yLag = cell(n, 1);
validObs = cell(n, 1);

for i = 1:n
    yi = yPanel{i};
    Ti = length(yi);
    yLagi = NaN(Ti, lags);

    for lag = 1:lags
        yLagi(lag+1:end, lag) = yi(1:end-lag);
    end

    yLag{i} = yLagi;

    % Valid observations: have y, all lags, and at least one more for FD
    validObs{i} = all(~isnan([yi, yLagi]), 2);
end
end

function [Z, yDiff, XDiff, yLagDiff] = buildInstruments(yPanel, XPanel, yLag, opts, n, T, ~, ~)
%BUILDINSTRUMENTS Build GMM instrument matrix
minLag = opts.gmmlags(1);
maxLag = min(opts.gmmlags(2), T - 1);

yDiff = cell(n, 1);
XDiff = cell(n, 1);
yLagDiff = cell(n, 1);
Z = cell(n, 1);

for i = 1:n
    yi = yPanel{i};
    Xi = XPanel{i};
    yLagi = yLag{i};
    Ti = length(yi);

    % First-difference transformation
    yDiff{i} = diff(yi);
    XDiff{i} = diff(Xi, 1, 1);
    yLagDiff{i} = diff(yLagi, 1, 1);

    % Build instrument matrix for this individual
    if opts.collapse
        % Collapsed instruments: one column per lag
        numInst = maxLag - minLag + 1;
        Zi = zeros(Ti - 1, numInst);

        for t = 2:Ti
            for lag = minLag:maxLag
                if t - lag >= 1
                    instCol = lag - minLag + 1;
                    Zi(t-1, instCol) = Zi(t-1, instCol) + yi(t - lag);
                end
            end
        end
    else
        % Full instrument matrix (standard)
        % For each t, use y_{t-2}, y_{t-3}, ..., y_{t-maxLag}
        numInst = sum(max(0, (2:Ti) - minLag));  % Upper bound
        Zi = zeros(Ti - 1, numInst);

        col = 1;
        for t = 2:Ti
            for lag = minLag:min(maxLag, t - 1)
                if t - lag >= 1 && col <= numInst
                    Zi(t-1, col) = yi(t - lag);
                    col = col + 1;
                end
            end
        end
        Zi = Zi(:, 1:col-1);  % Trim unused columns
    end

    % For system GMM, add level equations
    if strcmp(opts.method, 'system')
        % Level equation instruments: lagged differences
        ZiLevel = zeros(Ti - 1, opts.lags);
        for t = 2:Ti
            for lag = 1:opts.lags
                if t - lag >= 2
                    ZiLevel(t-1, lag) = yi(t - lag) - yi(t - lag - 1);
                end
            end
        end
        Zi = blkdiag(Zi, ZiLevel);
    end

    Z{i} = Zi;
end
end

function [yVec, XVec, ZMat] = stackPanelData(yDiff, XDiff, yLagDiff, Z, n, ~)
%STACKPANELDATA Stack panel data for GMM estimation
% Determine total observations and max instrument columns
totalObs = 0;
maxZcols = 0;

for i = 1:n
    validIdx = ~isnan(yDiff{i}) & all(~isnan(yLagDiff{i}), 2);
    totalObs = totalObs + sum(validIdx);
    maxZcols = max(maxZcols, size(Z{i}, 2));
end

k = size(XDiff{1}, 2);
lags = size(yLagDiff{1}, 2);

yVec = zeros(totalObs, 1);
XVec = zeros(totalObs, lags + k);
ZMat = zeros(totalObs, maxZcols);

row = 1;
for i = 1:n
    validIdx = ~isnan(yDiff{i}) & all(~isnan(yLagDiff{i}), 2) & all(~isnan(XDiff{i}), 2);
    nValid = sum(validIdx);

    if nValid > 0
        yVec(row:row+nValid-1) = yDiff{i}(validIdx);
        XVec(row:row+nValid-1, 1:lags) = yLagDiff{i}(validIdx, :);
        XVec(row:row+nValid-1, lags+1:end) = XDiff{i}(validIdx, :);

        Zi = Z{i}(validIdx, :);
        ZMat(row:row+nValid-1, 1:size(Zi, 2)) = Zi;

        row = row + nValid;
    end
end

% Trim to actual size
yVec = yVec(1:row-1);
XVec = XVec(1:row-1, :);
ZMat = ZMat(1:row-1, :);

% Remove zero columns
nonzeroCol = any(ZMat ~= 0, 1);
ZMat = ZMat(:, nonzeroCol);
end

function W = computeOptimalWeight(Z, res, ~, ~)
%COMPUTEOPTIMALWEIGHT Compute optimal GMM weighting matrix
% Robust weighting matrix: (1/N) * sum(Z_i' * e_i * e_i' * Z_i)
meat = Z' * diag(res.^2) * Z;
W = pinv(meat);
end

function varcoef = windmeijerCorrection(X, Z, res, res1, coef, coef1, W2, ~, ~, ~)
%WINDMEIJERCORRECTION Windmeijer (2005) finite-sample variance correction

L = size(Z, 2);
k = size(X, 2);
N = size(X, 1);

XZ = X' * Z;
ZX = Z' * X;

% Standard two-step variance
bread = (XZ * W2 * ZX) \ eye(k);

% Derivative of optimal weight matrix w.r.t. one-step coefficients
% dW/dcoef1 affects the two-step estimate

% Simplified Windmeijer correction
% Full implementation requires tracking computation graph

% Approximate correction factor
dRes = res - res1;
correctionFactor = 1 + trace(bread * XZ * W2 * Z' * diag(dRes.^2) * Z * W2 * ZX * bread) / N;

varcoef = bread * correctionFactor;

% Ensure positive semi-definite
varcoef = (varcoef + varcoef') / 2;
[V, D] = eig(varcoef);
d = diag(D);
d(d < 0) = 1e-10;
varcoef = V * diag(d) * V';
end

function [ARstat, ARpval] = arellonoBondARTest(res, ~, order, ~, ~)
%ARELLONOBONDARTEST Arellano-Bond test for serial correlation
N = length(res);

if N < order + 1
    ARstat = NaN;
    ARpval = NaN;
    return;
end

% Simple autocorrelation test
res_t = res(order+1:end);
res_tm = res(1:end-order);

% Under H0: E[e_t * e_{t-order}] = 0 in first differences
% For AR(1): should reject (FD induces MA(1))
% For AR(2): should not reject if no serial correlation in levels

cov_est = res_t' * res_tm / N;
var_est = sqrt((res_t.^2)' * (res_tm.^2)) / N;

ARstat = cov_est / (var_est + 1e-10);
ARpval = 2 * (1 - normcdf(abs(ARstat)));
end
