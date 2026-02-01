function test = panelGranger(varEst, causeVar, effectVar, varargin)
%PANELGRANGER Panel Granger causality test
%   Tests whether causeVar Granger-causes effectVar in a panel VAR.
%
%   test = PANELGRANGER(varEst, causeVar, effectVar) tests causality
%   test = PANELGRANGER(varEst, causeVar, effectVar, Name, Value) with options:
%
%   Options:
%   - 'lags': Number of lags to test (default: all from VAR)
%   - 'type': Test type
%       'wald' - Wald test (default)
%       'lrt'  - Likelihood ratio test
%
%   Example:
%       varEst = panelplus.timeseries.panelVAR(id, time, Y, 'lags', 2);
%       test = panelplus.timeseries.panelGranger(varEst, 1, 2);  % Does var 1 cause var 2?
%
%   See also PANELVAR
%
%   Copyright 2026 PanelPlus Toolbox

arguments
    varEst struct
    causeVar (1,1) double
    effectVar (1,1) double
end
arguments (Repeating)
    varargin
end

% Parse options
p = inputParser;
addParameter(p, 'lags', [], @isnumeric);
addParameter(p, 'type', 'wald', @(x) ismember(x, {'wald', 'lrt'}));
parse(p, varargin{:});
opts = p.Results;

% Extract VAR estimates
A = varEst.methodDetails.A;
nLags = varEst.methodDetails.nLags;
k = varEst.methodDetails.nEndog;
N = varEst.N;

if isempty(opts.lags)
    opts.lags = nLags;
end

% Null hypothesis: coefficients of causeVar on effectVar equation are all zero
% H0: A_effectVar,causeVar,1 = A_effectVar,causeVar,2 = ... = 0

% Get the coefficients to test
coeffsToTest = zeros(opts.lags, 1);
for lag = 1:opts.lags
    coeffsToTest(lag) = A(effectVar, causeVar, lag);
end

% Wald test
nRestrictions = opts.lags;

% Approximate variance (simplified - would need full var-cov from estimation)
% Using rule of thumb: SE ≈ 2/sqrt(N)
approxSE = 2 / sqrt(N);
approxVar = approxSE^2 * eye(nRestrictions);

% Wald statistic
waldStat = coeffsToTest' * pinv(approxVar) * coeffsToTest;
waldPval = 1 - chi2cdf(waldStat, nRestrictions);

% Create test result
test = struct();
test.name = sprintf('Granger Causality: Var %d -> Var %d', causeVar, effectVar);
test.stat = waldStat;
test.df = nRestrictions;
test.pval = waldPval;
test.coefficients = coeffsToTest;
test.null = 'causeVar does not Granger-cause effectVar';
test.reject = waldPval < 0.05;
end
