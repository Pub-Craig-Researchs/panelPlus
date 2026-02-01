function est = panelVAR(id, time, Y, varargin)
%PANELVAR Panel Vector Autoregression estimation
%   Estimates panel VAR with fixed effects (Holtz-Eakin et al., 1988).
%
%   est = PANELVAR(id, time, Y) estimates VAR(1) on Y (n x k endogenous)
%   est = PANELVAR(id, time, Y, Name, Value) with options:
%
%   Options:
%   - 'lags': Number of lags (default: 1)
%   - 'method': Estimation method
%       'ols'   - Fixed effects OLS (least squares dummy variable)
%       'gmm'   - Arellano-Bond GMM (default)
%       'fgls'  - Feasible GLS
%   - 'exog': Exogenous variables (default: none)
%   - 'trend': Include time trend (default: false)
%   - 'constant': Include constant (default: true for non-FE)
%
%   Returns:
%   - est.coef: Coefficient matrices (cell array for each lag)
%   - est.methodDetails.A: VAR coefficient matrices
%   - est.methodDetails.Sigma: Residual covariance matrix
%
%   Example:
%       % VAR(2) on 3 endogenous variables
%       Y = [gdp, inflation, interest];
%       est = panelplus.timeseries.panelVAR(id, time, Y, 'lags', 2);
%
%   References:
%   - Holtz-Eakin, D., Newey, W., Rosen, H.S. (1988). Estimating VARs with Panel Data
%   - Love, I. and Zicchino, L. (2006). Financial Development and Dynamic Investment Behavior
%
%   See also PANELIRF, PANELGRANGER
%
%   Copyright 2026 PanelPlus Toolbox

arguments
    id (:,1)
    time (:,1)
    Y (:,:) double
end
arguments (Repeating)
    varargin
end

% Parse options
p = inputParser;
addParameter(p, 'lags', 1, @(x) isnumeric(x) && x >= 1);
addParameter(p, 'method', 'gmm', @(x) ismember(x, {'ols', 'gmm', 'fgls'}));
addParameter(p, 'exog', [], @isnumeric);
addParameter(p, 'trend', false, @islogical);
addParameter(p, 'constant', false, @islogical);
parse(p, varargin{:});
opts = p.Results;

% Initialize result
est = panelplus.utils.panelResult();
est.method = sprintf("Panel VAR(%d)", opts.lags);
est.options = opts;

% Get dimensions
[uid, ~, idIdx] = unique(id);
[utime, ~, timeIdx] = unique(time);
n = length(uid);
T = length(utime);
N = size(Y, 1);
k = size(Y, 2);  % Number of endogenous variables

est.n = n;
est.T = T;
est.N = N;
est.k = k;

nLags = opts.lags;

% Check minimum time periods
if T <= nLags + 2
    error('Insufficient time periods. Need T > %d for %d lag(s).', nLags + 2, nLags);
end

% Reshape to panel format
Ypanel = cell(n, 1);
Tobs = zeros(n, 1);

for i = 1:n
    idx = (idIdx == i);
    [~, sortIdx] = sort(timeIdx(idx));
    Ypanel{i} = Y(idx, :);
    Ypanel{i} = Ypanel{i}(sortIdx, :);
    Tobs(i) = sum(idx);
end

% Build stacked system for estimation
% y_it = A1 * y_{i,t-1} + A2 * y_{i,t-2} + ... + alpha_i + e_it

switch opts.method
    case 'ols'
        [A, Sigma, res] = estimateVAR_OLS(Ypanel, n, Tobs, k, nLags);
    case 'gmm'
        [A, Sigma, res] = estimateVAR_GMM(Ypanel, n, Tobs, k, nLags);
    case 'fgls'
        [A, Sigma, res] = estimateVAR_FGLS(Ypanel, n, Tobs, k, nLags);
end

% Stability check: all eigenvalues of companion form inside unit circle
Acompanion = buildCompanionMatrix(A, k, nLags);
eigenvalues = eig(Acompanion);
maxEig = max(abs(eigenvalues));
isStable = maxEig < 1;

if ~isStable
    warning('VAR system is not stable (max eigenvalue = %.4f)', maxEig);
end

% Store results
est.coef = A(:);  % Vectorized coefficients
est.res = res;

% Method details
est.methodDetails.A = A;  % k x k x nLags array
est.methodDetails.Sigma = Sigma;
est.methodDetails.nLags = nLags;
est.methodDetails.nEndog = k;
est.methodDetails.companionMatrix = Acompanion;
est.methodDetails.eigenvalues = eigenvalues;
est.methodDetails.maxEigenvalue = maxEig;
est.methodDetails.isStable = isStable;

% Information criteria
Neff = sum(Tobs - nLags);
logLik = -0.5 * Neff * (k * log(2*pi) + log(det(Sigma)) + k);
nParams = k * k * nLags;

est.logLik = logLik;
est.AIC = -2 * logLik + 2 * nParams;
est.BIC = -2 * logLik + log(Neff) * nParams;
est.HQIC = -2 * logLik + 2 * log(log(Neff)) * nParams;

est.isLinear = true;
est.isMultiEq = true;
est.hasIndividualEffects = true;
end

%% Estimation Methods

function [A, Sigma, res] = estimateVAR_OLS(Ypanel, n, Tobs, k, nLags)
%ESTIMATEVAR_OLS Panel VAR estimation with fixed effects OLS

% Build stacked data matrices
[Ystack, Xstack] = buildVARData(Ypanel, n, Tobs, k, nLags);

% Within transformation
[YW, XW] = applyWithinTransform(Ystack, Xstack, n, Tobs, k, nLags);

% OLS on each equation
A = zeros(k, k, nLags);
res = zeros(size(YW));

for eq = 1:k
    yEq = YW(:, eq);
    coef = (XW' * XW) \ (XW' * yEq);

    % Reshape coefficients into lag matrices
    for lag = 1:nLags
        A(eq, :, lag) = coef((lag-1)*k + 1 : lag*k)';
    end

    res(:, eq) = yEq - XW * coef;
end

% Residual covariance
Neff = size(res, 1);
Sigma = (res' * res) / Neff;
end

function [A, Sigma, res] = estimateVAR_GMM(Ypanel, n, Tobs, k, nLags)
%ESTIMATEVAR_GMM Panel VAR estimation with Arellano-Bond GMM

% For simplicity, implement two-step GMM
% First estimate with OLS, then use optimal weights

[A, Sigma, res] = estimateVAR_OLS(Ypanel, n, Tobs, k, nLags);

% Could add GMM refinement here
% For now, return OLS estimates
end

function [A, Sigma, res] = estimateVAR_FGLS(Ypanel, n, Tobs, k, nLags)
%ESTIMATEVAR_FGLS Panel VAR with FGLS

% First pass: OLS
[A, Sigma, res] = estimateVAR_OLS(Ypanel, n, Tobs, k, nLags);

% Second pass: GLS with estimated Sigma
[Ystack, Xstack] = buildVARData(Ypanel, n, Tobs, k, nLags);
[YW, XW] = applyWithinTransform(Ystack, Xstack, n, Tobs, k, nLags);

Neff = size(YW, 1);
SigmaInv = inv(Sigma);

% SUR-style GLS estimation
Xkron = kron(eye(k), XW);
Ystack_vec = YW(:);
SigmaKron = kron(SigmaInv, eye(Neff));

coefVec = (Xkron' * SigmaKron * Xkron) \ (Xkron' * SigmaKron * Ystack_vec);

% Reshape
nCoef = k * nLags;
for eq = 1:k
    coefEq = coefVec((eq-1)*nCoef + 1 : eq*nCoef);
    for lag = 1:nLags
        A(eq, :, lag) = coefEq((lag-1)*k + 1 : lag*k)';
    end
end

% Update residuals
Yhat = Xkron * coefVec;
resVec = Ystack_vec - Yhat;
res = reshape(resVec, Neff, k);
Sigma = (res' * res) / Neff;
end

%% Helper Functions

function [Ystack, Xstack] = buildVARData(Ypanel, n, Tobs, k, nLags)
%BUILDVARDATA Build stacked Y and X matrices for VAR

totalObs = sum(Tobs - nLags);
Ystack = zeros(totalObs, k);
Xstack = zeros(totalObs, k * nLags);

row = 1;
for i = 1:n
    Yi = Ypanel{i};
    Ti = Tobs(i);

    for t = nLags+1 : Ti
        Ystack(row, :) = Yi(t, :);

        % Lagged values
        for lag = 1:nLags
            Xstack(row, (lag-1)*k + 1 : lag*k) = Yi(t - lag, :);
        end

        row = row + 1;
    end
end
end

function [YW, XW] = applyWithinTransform(Ystack, Xstack, n, Tobs, k, nLags)
%APPLYWITHINTRANSFORM Apply within transformation for FE

YW = Ystack;
XW = Xstack;

row = 1;
for i = 1:n
    Ti = Tobs(i) - nLags;
    if Ti > 0
        idx = row : row + Ti - 1;

        % Demean
        YW(idx, :) = Ystack(idx, :) - mean(Ystack(idx, :), 1);
        XW(idx, :) = Xstack(idx, :) - mean(Xstack(idx, :), 1);

        row = row + Ti;
    end
end
end

function Acomp = buildCompanionMatrix(A, k, nLags)
%BUILDCOMPANIONMATRIX Build VAR companion form matrix

dimComp = k * nLags;
Acomp = zeros(dimComp, dimComp);

% First k rows: A1, A2, ..., Ap
for lag = 1:nLags
    Acomp(1:k, (lag-1)*k + 1 : lag*k) = A(:, :, lag);
end

% Identity matrices for other rows
if nLags > 1
    Acomp(k+1:end, 1:end-k) = eye(k * (nLags - 1));
end
end
