function est = panelFGLS(id, time, y, X, varargin)
%PANELFGLS Panel Feasible Generalized Least Squares estimation
%   Computes FGLS estimation for panel data with various error structures.
%
%   est = PANELFGLS(id, time, y, X) computes FGLS with default settings
%   est = PANELFGLS(id, time, y, X, Name, Value) with additional options
%
%   Options:
%   - 'arType': Error AR structure
%       'none'  - No serial correlation (default)
%       'ar1'   - Common AR(1) across panels
%       'psar1' - Panel-specific AR(1)
%   - 'panels': Cross-sectional correlation structure
%       'homo'       - Homoskedastic errors (default)
%       'hetero'     - Heteroskedastic (panel-specific variances)
%       'correlated' - Cross-sectionally correlated (SUR)
%   - 'method': Estimation approach
%       'fe' - Fixed effects (within transformation first)
%       're' - Random effects
%       'po' - Pooled
%   - 'maxIter': Maximum iterations for FGLS (default: 100)
%   - 'tol': Convergence tolerance (default: 1e-6)
%
%   Example:
%       est = panelplus.estimation.panelFGLS(id, time, y, X, ...
%                   'arType', 'ar1', 'panels', 'hetero');
%
%   References:
%   - Parks (1967) - AR(1) GLS
%   - Beck and Katz (1995) - PCSE
%   - Wooldridge (2010) - Econometric Analysis of Cross Section and Panel Data
%
%   See also PANEL, PANELGMM
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
addParameter(p, 'arType', 'none', @(x) ismember(x, {'none', 'ar1', 'psar1'}));
addParameter(p, 'panels', 'homo', @(x) ismember(x, {'homo', 'hetero', 'correlated'}));
addParameter(p, 'method', 'fe', @(x) ismember(x, {'fe', 're', 'po'}));
addParameter(p, 'maxIter', 100, @isnumeric);
addParameter(p, 'tol', 1e-6, @isnumeric);
addParameter(p, 'dfCorrection', true, @islogical);
parse(p, varargin{:});
opts = p.Results;

% Initialize result
est = panelplus.utils.panelResult();
est.method = "Panel FGLS";
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
est.k = k;

% Check if balanced
est.isBalanced = (n * T == N);

% Add constant if needed
hasConstant = ~any(all(X == 1, 1));
if hasConstant && strcmp(opts.method, 'po')
    X = [X, ones(N, 1)];
    k = k + 1;
    est.k = k;
end
est.hasConstant = hasConstant;

% =====================
% Step 1: Initial OLS
% =====================
if strcmp(opts.method, 'fe')
    % Within transformation
    [yW, XW] = withinTransform(y, X, idIdx, n);
    coef0 = (XW' * XW) \ (XW' * yW);
    res0 = yW - XW * coef0;
else
    coef0 = (X' * X) \ (X' * y);
    res0 = y - X * coef0;
end

% =====================
% Step 2: Estimate error covariance structure
% =====================

% Reshape residuals to panel format (n x T)
if est.isBalanced
    resPanel = reshape(res0, T, n)';  % n x T matrix
else
    % For unbalanced, create sparse representation
    resPanel = NaN(n, T);
    for i = 1:N
        resPanel(idIdx(i), timeIdx(i)) = res0(i);
    end
end

% Estimate AR coefficient(s)
rho = estimateARCoef(resPanel, opts.arType, n, T);

% Estimate cross-sectional covariance
Omega = estimateCrossSectionalCov(resPanel, opts.panels, n, T);

% =====================
% Step 3: FGLS Estimation
% =====================

% Build full variance-covariance matrix
SigmaInv = buildSigmaInverse(rho, Omega, opts, n, T, idIdx, timeIdx, N);

% FGLS estimation
if strcmp(opts.method, 'fe')
    XtSinvX = XW' * SigmaInv * XW;
    XtSinvy = XW' * SigmaInv * yW;
else
    XtSinvX = X' * SigmaInv * X;
    XtSinvy = X' * SigmaInv * y;
end

coef = XtSinvX \ XtSinvy;

% Variance-covariance of coefficients
varcoef = inv(XtSinvX);

% Residuals
if strcmp(opts.method, 'fe')
    res = yW - XW * coef;
    yhat = XW * coef;
else
    res = y - X * coef;
    yhat = X * coef;
end

% Standard errors
stderr = sqrt(diag(varcoef));

% t-statistics and p-values
tstat = coef ./ stderr;
resdf = N - k - (strcmp(opts.method, 'fe') * (n - 1));
pval = 2 * (1 - tcdf(abs(tstat), resdf));

% Goodness of fit
RSS = res' * res;
if strcmp(opts.method, 'fe')
    TSS = yW' * yW;
else
    yMean = mean(y);
    TSS = (y - yMean)' * (y - yMean);
end
ESS = TSS - RSS;
r2 = 1 - RSS / TSS;
adjr2 = 1 - (1 - r2) * (N - 1) / (N - k);

% =====================
% Store results
% =====================
est.coef = coef;
est.varcoef = varcoef;
est.stderr = stderr;
est.tstat = tstat;
est.pval = pval;
est.yhat = yhat;
est.res = res;
est.resvar = RSS / resdf;
est.resdf = resdf;
est.RSS = RSS;
est.ESS = ESS;
est.TSS = TSS;
est.r2 = r2;
est.adjr2 = adjr2;

% Method-specific details
est.methodDetails.rho = rho;
est.methodDetails.Omega = Omega;
est.methodDetails.arType = opts.arType;
est.methodDetails.panelStructure = opts.panels;

est.hasIndividualEffects = strcmp(opts.method, 'fe');
est.isLinear = true;
est.isAsymptotic = true;
end

%% Helper Functions

function [yW, XW] = withinTransform(y, X, idIdx, n)
%WITHINTRANSFORM Apply within (demeaning) transformation
N = length(y);
k = size(X, 2);

yW = y;
XW = X;

for i = 1:n
    idx = (idIdx == i);
    yW(idx) = y(idx) - mean(y(idx));
    for j = 1:k
        XW(idx, j) = X(idx, j) - mean(X(idx, j));
    end
end
end

function rho = estimateARCoef(resPanel, arType, n, ~)
%ESTIMATEARCOEF Estimate AR(1) coefficient(s) from residual panel
switch arType
    case 'none'
        rho = 0;
    case 'ar1'
        % Common AR(1): pool all panels
        res_t = resPanel(:, 2:end);
        res_tm1 = resPanel(:, 1:end-1);

        % Remove NaN for unbalanced panels
        validIdx = ~isnan(res_t) & ~isnan(res_tm1);
        rho = sum(res_t(validIdx) .* res_tm1(validIdx)) / ...
            sum(res_tm1(validIdx).^2);

    case 'psar1'
        % Panel-specific AR(1)
        rho = zeros(n, 1);
        for i = 1:n
            res_i = resPanel(i, :);
            validT = find(~isnan(res_i));
            if length(validT) > 1
                res_t = res_i(validT(2:end));
                res_tm1 = res_i(validT(1:end-1));
                rho(i) = sum(res_t .* res_tm1) / sum(res_tm1.^2);
            end
        end
end
end

function Omega = estimateCrossSectionalCov(resPanel, panelType, n, ~)
%ESTIMATECROSSSECTIONALCOV Estimate cross-sectional error covariance
switch panelType
    case 'homo'
        % Homoskedastic: scalar variance
        validRes = resPanel(~isnan(resPanel));
        Omega = var(validRes) * eye(n);

    case 'hetero'
        % Heteroskedastic: diagonal with panel-specific variances
        sigma2 = zeros(n, 1);
        for i = 1:n
            res_i = resPanel(i, ~isnan(resPanel(i, :)));
            sigma2(i) = var(res_i);
        end
        Omega = diag(sigma2);

    case 'correlated'
        % Full cross-sectional correlation (SUR-style)
        % Handle missing values by pairwise computation
        Omega = zeros(n, n);
        for i = 1:n
            for j = i:n
                res_i = resPanel(i, :);
                res_j = resPanel(j, :);
                validT = ~isnan(res_i) & ~isnan(res_j);
                if sum(validT) > 0
                    Omega(i, j) = mean(res_i(validT) .* res_j(validT));
                    Omega(j, i) = Omega(i, j);
                end
            end
        end
end
end

function SigmaInv = buildSigmaInverse(rho, Omega, opts, n, T, idIdx, timeIdx, N)
%BUILDSIGMAINVERSE Build and invert the full error covariance matrix

if strcmp(opts.arType, 'none') && strcmp(opts.panels, 'homo')
    % Simple case: identity matrix (scaled)
    SigmaInv = speye(N);
    return;
end

% For balanced panels with common AR(1)
if strcmp(opts.arType, 'ar1') || strcmp(opts.arType, 'none')
    rhoVal = rho(1);

    % AR(1) inverse for each time series (T x T block)
    if abs(rhoVal) < 1e-10
        PhiInv = eye(T);
    else
        PhiInv = zeros(T, T);
        PhiInv(1, 1) = 1;
        PhiInv(T, T) = 1;
        for t = 2:T-1
            PhiInv(t, t) = 1 + rhoVal^2;
        end
        for t = 1:T-1
            PhiInv(t, t+1) = -rhoVal;
            PhiInv(t+1, t) = -rhoVal;
        end
        PhiInv = PhiInv / (1 - rhoVal^2);
    end
end

% Cross-sectional inverse
OmegaInv = inv(Omega);

% Kronecker product for balanced panels
% Sigma = Omega ⊗ Phi => Sigma^(-1) = Omega^(-1) ⊗ Phi^(-1)
if n * T == N  % Balanced
    SigmaInv = kron(OmegaInv, PhiInv);
else
    % For unbalanced panels, build block-diagonal structure
    SigmaInv = sparse(N, N);
    for i = 1:n
        idx_i = find(idIdx == i);
        Ti = length(idx_i);

        if strcmp(opts.arType, 'psar1')
            rhoVal = rho(i);
        else
            rhoVal = rho(1);
        end

        % Build Ti x Ti AR inverse block
        if Ti == 1
            PhiInv_i = 1;
        else
            PhiInv_i = buildARInverse(rhoVal, Ti);
        end

        % Scale by cross-sectional variance
        SigmaInv(idx_i, idx_i) = PhiInv_i / Omega(i, i);
    end
end
end

function PhiInv = buildARInverse(rho, T)
%BUILDARINVERSE Build inverse of AR(1) covariance matrix
if abs(rho) < 1e-10
    PhiInv = eye(T);
    return;
end

PhiInv = zeros(T, T);
PhiInv(1, 1) = 1;
PhiInv(T, T) = 1;
for t = 2:T-1
    PhiInv(t, t) = 1 + rho^2;
end
for t = 1:T-1
    PhiInv(t, t+1) = -rho;
    PhiInv(t+1, t) = -rho;
end
PhiInv = PhiInv / (1 - rho^2);
end
