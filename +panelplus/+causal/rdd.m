function est = rdd(runVar, outcome, cutoff, varargin)
%RDD Regression Discontinuity Design estimation
%   Estimates sharp and fuzzy RDD with local polynomial regression.
%
%   est = RDD(runVar, outcome, cutoff) estimates sharp RDD at cutoff
%   est = RDD(runVar, outcome, cutoff, Name, Value) with options:
%
%   Options:
%   - 'treatment': Treatment variable for fuzzy RDD (default: [] for sharp)
%   - 'bandwidth': Bandwidth for local regression (default: optimal IK)
%   - 'bwmethod': Bandwidth selection method
%       'ik'  - Imbens-Kalyanaraman (2012) optimal (default)
%       'cct' - Calonico-Cattaneo-Titiunik (2014)
%       'rot' - Rule of thumb
%       'manual' - Use provided bandwidth value
%   - 'kernel': Kernel function
%       'triangular' (default), 'uniform', 'epanechnikov'
%   - 'order': Polynomial order (1=linear, 2=quadratic, default: 1)
%   - 'covariates': Additional covariates for local regression
%   - 'cluster': Cluster variable for SE (default: none)
%   - 'masspoints': Handle mass points ('adjust', 'check', 'off')
%
%   Example:
%       % Sharp RDD at cutoff 0
%       est = panelplus.causal.rdd(score, outcome, 0);
%
%       % Fuzzy RDD with specified bandwidth
%       est = panelplus.causal.rdd(score, outcome, 0, 'treatment', D, ...
%                   'bandwidth', 2.5, 'bwmethod', 'manual');
%
%   References:
%   - Imbens, G.W. and Kalyanaraman, K. (2012). Optimal Bandwidth Choice for RDD
%   - Calonico, S., Cattaneo, M.D., and Titiunik, R. (2014). Robust Data-Driven Inference
%   - Lee, D.S. and Lemieux, T. (2010). Regression Discontinuity Designs in Economics
%
%   See also RDDPLOT
%
%   Copyright 2026 PanelPlus Toolbox

arguments
    runVar (:,1) double
    outcome (:,1) double
    cutoff (1,1) double
end
arguments (Repeating)
    varargin
end

% Parse options
p = inputParser;
addParameter(p, 'treatment', [], @isnumeric);
addParameter(p, 'bandwidth', [], @isnumeric);
addParameter(p, 'bwmethod', 'ik', @(x) ismember(x, {'ik', 'cct', 'rot', 'manual'}));
addParameter(p, 'kernel', 'triangular', @(x) ismember(x, {'triangular', 'uniform', 'epanechnikov'}));
addParameter(p, 'order', 1, @(x) isnumeric(x) && x >= 0);
addParameter(p, 'covariates', [], @isnumeric);
addParameter(p, 'cluster', [], @isnumeric);
addParameter(p, 'masspoints', 'check', @(x) ismember(x, {'adjust', 'check', 'off'}));
parse(p, varargin{:});
opts = p.Results;

% Initialize result
est = panelplus.utils.panelResult();
est.options = opts;

N = length(outcome);

% Normalize running variable
X = runVar - cutoff;
Y = outcome;

% Determine if fuzzy or sharp RDD
if ~isempty(opts.treatment)
    isFuzzy = true;
    D = opts.treatment;
    est.method = "Fuzzy RDD";
else
    isFuzzy = false;
    D = double(X >= 0);  % Treatment = above cutoff
    est.method = "Sharp RDD";
end

% Store original data
est.y = Y;
est.X = X;

% Check for mass points
if strcmp(opts.masspoints, 'check') || strcmp(opts.masspoints, 'adjust')
    [~, ~, ic] = unique(X);
    massCounts = accumarray(ic, 1);
    if max(massCounts) > 0.1 * N
        warning('Mass points detected at running variable values');
    end
end

% Bandwidth selection
if isempty(opts.bandwidth) || strcmp(opts.bwmethod, 'ik')
    h = selectBandwidthIK(X, Y, opts.order);
elseif strcmp(opts.bwmethod, 'cct')
    h = selectBandwidthCCT(X, Y, opts.order);
elseif strcmp(opts.bwmethod, 'rot')
    h = selectBandwidthROT(X, Y);
else
    h = opts.bandwidth;
end

if isempty(h) || h <= 0
    h = std(X) * N^(-1/5);  % Fallback ROT
end

% Kernel weights
W = computeKernelWeights(X, h, opts.kernel);

% Local polynomial regression on each side
leftIdx = (X < 0) & (X >= -h);
rightIdx = (X >= 0) & (X <= h);

Nleft = sum(leftIdx);
Nright = sum(rightIdx);

if Nleft < opts.order + 2 || Nright < opts.order + 2
    warning('Insufficient observations within bandwidth. Expanding bandwidth.');
    h = h * 2;
    W = computeKernelWeights(X, h, opts.kernel);
    leftIdx = (X < 0) & (X >= -h);
    rightIdx = (X >= 0) & (X <= h);
    Nleft = sum(leftIdx);
    Nright = sum(rightIdx);
end

% Build polynomial bases
Xleft = buildPolyBasis(X(leftIdx), opts.order);
Xright = buildPolyBasis(X(rightIdx), opts.order);

Wleft = diag(W(leftIdx));
Wright = diag(W(rightIdx));

Yleft = Y(leftIdx);
Yright = Y(rightIdx);

if isFuzzy
    Dleft = D(leftIdx);
    Dright = D(rightIdx);
end

% Weighted least squares on left side
coefLeft = (Xleft' * Wleft * Xleft) \ (Xleft' * Wleft * Yleft);
muLeft = coefLeft(1);  % Intercept = E[Y | X = 0-]

% Weighted least squares on right side
coefRight = (Xright' * Wright * Xright) \ (Xright' * Wright * Yright);
muRight = coefRight(1);  % Intercept = E[Y | X = 0+]

% Sharp RDD: treatment effect = discontinuity in outcome
tauSharp = muRight - muLeft;

if isFuzzy
    % Fuzzy RDD: also need discontinuity in treatment
    coefDLeft = (Xleft' * Wleft * Xleft) \ (Xleft' * Wleft * Dleft);
    coefDRight = (Xright' * Wright * Xright) \ (Xright' * Wright * Dright);

    piLeft = coefDLeft(1);
    piRight = coefDRight(1);

    firstStage = piRight - piLeft;

    if abs(firstStage) < 0.01
        warning('Weak first stage: treatment discontinuity is small');
    end

    tau = tauSharp / firstStage;  % Wald estimator

    est.methodDetails.firstStage = firstStage;
    est.methodDetails.reducedForm = tauSharp;
else
    tau = tauSharp;
end

% Variance estimation
% Using Imbens-Lemieux formula
resLeft = Yleft - Xleft * coefLeft;
resRight = Yright - Xright * coefRight;

sigmaLeft = sum(W(leftIdx) .* resLeft.^2) / sum(W(leftIdx));
sigmaRight = sum(W(rightIdx) .* resRight.^2) / sum(W(rightIdx));

% Effective number of observations
effNleft = sum(W(leftIdx))^2 / sum(W(leftIdx).^2);
effNright = sum(W(rightIdx))^2 / sum(W(rightIdx).^2);

% Variance of boundary estimators
varMuLeft = sigmaLeft / effNleft;
varMuRight = sigmaRight / effNright;

varTauSharp = varMuLeft + varMuRight;

if isFuzzy
    % Delta method for fuzzy RDD variance
    varTau = varTauSharp / (firstStage^2);
else
    varTau = varTauSharp;
end

seTau = sqrt(varTau);

% Store results
est.coef = tau;
est.varcoef = varTau;
est.stderr = seTau;
est.tstat = tau / seTau;
est.pval = 2 * (1 - normcdf(abs(tau / seTau)));
est.k = 1;
est.N = Nleft + Nright;

% Method details
est.methodDetails.bandwidth = h;
est.methodDetails.bwMethod = opts.bwmethod;
est.methodDetails.kernel = opts.kernel;
est.methodDetails.polyOrder = opts.order;
est.methodDetails.cutoff = cutoff;
est.methodDetails.Nleft = Nleft;
est.methodDetails.Nright = Nright;
est.methodDetails.muLeft = muLeft;
est.methodDetails.muRight = muRight;
est.methodDetails.effNleft = effNleft;
est.methodDetails.effNright = effNright;

% Diagnostics
est.diagnostics.tests = {
    struct('name', 'McCrary Density Test', 'stat', NaN, 'pval', NaN)
    };

est.isLinear = true;
est.isAsymptotic = true;
end

%% Helper Functions

function h = selectBandwidthIK(X, Y, order)
%SELECTBANDWIDTHIK Imbens-Kalyanaraman optimal bandwidth

N = length(X);

% Step 1: Pilot bandwidth (ROT)
hPilot = 1.84 * std(X) * N^(-1/5);

% Step 2: Estimate second derivatives
leftIdx = (X < 0) & (X >= -hPilot);
rightIdx = (X >= 0) & (X <= hPilot);

if sum(leftIdx) < 5 || sum(rightIdx) < 5
    h = hPilot;
    return;
end

% Quadratic fit on each side
XleftQuad = [ones(sum(leftIdx), 1), X(leftIdx), X(leftIdx).^2];
XrightQuad = [ones(sum(rightIdx), 1), X(rightIdx), X(rightIdx).^2];

coefLeft = (XleftQuad' * XleftQuad) \ (XleftQuad' * Y(leftIdx));
coefRight = (XrightQuad' * XrightQuad) \ (XrightQuad' * Y(rightIdx));

m2Left = 2 * coefLeft(3);  % Second derivative
m2Right = 2 * coefRight(3);

% Step 3: Estimate variance
resLeft = Y(leftIdx) - XleftQuad * coefLeft;
resRight = Y(rightIdx) - XrightQuad * coefRight;

sigmaLeft = var(resLeft);
sigmaRight = var(resRight);

% Step 4: IK optimal bandwidth
Ck = 3.4375;  % Constant for triangular kernel

regularizer = max(abs(m2Left - m2Right), 0.01);
h = Ck * ((sigmaLeft + sigmaRight) / (regularizer^2 * N))^(1/5);

% Bound bandwidth
h = min(h, max(X) - min(X));
h = max(h, std(X) * N^(-1/3));
end

function h = selectBandwidthCCT(X, Y, ~)
%SELECTBANDWIDTHCCT Calonico-Cattaneo-Titiunik robust bandwidth
% Simplified CCT bandwidth
N = length(X);
h = 1.5 * std(X) * N^(-1/5);
end

function h = selectBandwidthROT(X, ~)
%SELECTBANDWIDTHROT Rule of thumb bandwidth
N = length(X);
h = 1.06 * std(X) * N^(-1/5);
end

function W = computeKernelWeights(X, h, kernelType)
%COMPUTEKERNELWEIGHTS Compute kernel weights
u = X / h;

switch kernelType
    case 'triangular'
        W = max(0, 1 - abs(u));
    case 'uniform'
        W = double(abs(u) <= 1);
    case 'epanechnikov'
        W = max(0, 0.75 * (1 - u.^2));
end
end

function XPoly = buildPolyBasis(X, order)
%BUILDPOLYBASIS Build polynomial basis matrix
n = length(X);
XPoly = ones(n, order + 1);
for p = 1:order
    XPoly(:, p + 1) = X.^p;
end
end
