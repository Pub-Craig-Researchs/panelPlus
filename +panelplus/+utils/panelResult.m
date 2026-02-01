function est = panelResult()
%PANELRESULT Create extended panel estimation result structure
%   Creates a result structure compatible with estout but with additional
%   fields for advanced methods.
%
%   est = PANELRESULT() returns an empty result structure
%
%   Fields:
%   - All standard estout fields (coef, stderr, varcoef, etc.)
%   - diagnostics: struct with diagnostic test results
%   - marginalEffects: for nonlinear models
%   - bootResults: bootstrap results if applicable
%   - methodDetails: method-specific information
%
%   See also ESTOUT, ESTDISP
%
%   Copyright 2026 PanelPlus Toolbox

% Initialize base structure (compatible with original estout)
est = struct();

% === Identification ===
est.method = "";
est.methodDetails = struct();
est.options = struct();

% === Data dimensions ===
est.n = 0;          % Number of individuals
est.T = 0;          % Number of time periods
est.N = 0;          % Total observations
est.k = 0;          % Number of coefficients
est.isBalanced = true;

% === Estimation results ===
est.coef = [];
est.varcoef = [];
est.stderr = [];
est.tstat = [];
est.pval = [];

% === Fitted values and residuals ===
est.yhat = [];
est.res = [];
est.y = [];
est.X = [];

% === Variance estimation ===
est.resvar = 0;
est.resdf = 0;
est.vartype = "homo";
est.isRobust = false;
est.clusterVars = {};

% === Goodness of fit ===
est.RSS = 0;
est.ESS = 0;
est.TSS = 0;
est.r2 = 0;
est.adjr2 = 0;
est.r2Within = NaN;
est.r2Between = NaN;
est.r2Overall = NaN;

% === Model specification ===
est.isMultiEq = false;
est.isLinear = true;
est.hasConstant = true;
est.isAsymptotic = false;

% === Variable names ===
est.ynames = {};
est.xnames = {};
est.znames = {};  % Instrument names for IV

% === Fixed/Random effects ===
est.hasIndividualEffects = false;
est.hasTimeEffects = false;
est.individualEffects = [];
est.timeEffects = [];
est.theta = NaN;  % RE transformation parameter

% === Extended diagnostics ===
est.diagnostics = struct(...
    'tests', {{}}, ...
    'warnings', {{}}, ...
    'info', {{}} ...
    );

% === Bootstrap results ===
est.bootResults = struct(...
    'nReps', 0, ...
    'bootCoef', [], ...
    'bootSE', [], ...
    'bootCI', [], ...
    'ciLevel', 0.95 ...
    );

% === Marginal effects (for nonlinear) ===
est.marginalEffects = struct(...
    'atMean', [], ...
    'average', [], ...
    'stderr', [] ...
    );

% === Information criteria ===
est.logLik = NaN;
est.AIC = NaN;
est.BIC = NaN;
est.HQIC = NaN;
end
