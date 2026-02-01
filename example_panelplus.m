%% PanelPlus Toolbox - Comprehensive Example
%  Demonstrates all methods in the PanelPlus package
%
%  This script shows how to use:
%  - Multi-way clustered standard errors
%  - Panel FGLS estimation
%  - Panel GMM (Arellano-Bond, Blundell-Bond)
%  - Difference-in-Differences (DID)
%  - Staggered DID (Callaway-Sant'Anna)
%  - Event Study
%  - Regression Discontinuity Design (RDD)
%  - Panel VAR and Impulse Response Functions
%  - Double/Debiased Machine Learning (DDML)
%  - Causal Forest
%
%  Copyright 2026 PanelPlus Toolbox

clear; clc;

%% Add paths
% Add current directory for +panelplus package
addpath(pwd);  % This enables panelplus.* namespace

% Add original toolbox paths
addpath('PanelDataToolboxV2/paneldata/estimation');
addpath('PanelDataToolboxV2/paneldata/util');
addpath('PanelDataToolboxV2/paneldata/tests');
addpath('PanelDataToolboxV2/paneldata/stafun');
addpath('v76i06-data');

%% Load example data
load('MunnellData')

y = log(gsp);
X = [log(pcap), log(pc), log(emp), unemp];

fprintf('=== PanelPlus Toolbox Demo ===\n\n');
fprintf('Data: Munnell (1990) US States Production Data\n');
fprintf('  N = %d individuals (states)\n', length(unique(id)));
fprintf('  T = %d time periods\n', length(unique(year)));
fprintf('  Total observations: %d\n\n', length(y));

%% 1. Standard Panel FE with Multi-Way Clustering
fprintf('=== 1. Multi-Way Clustering Demo ===\n');

% Standard panel FE from original toolbox
fe = panel(id, year, y, X, 'fe');
fe.ynames = {'lgsp'};
fe.xnames = {'lpcap', 'lpc', 'lemp', 'unemp'};
estdisp(fe);

% Multi-way clustering with our new utility
% Simulate two-way clustering (id x year)
XFull = [X, dummyvar(categorical(id)), dummyvar(categorical(year))];
res = y - XFull * (XFull \ y);

varcoefMW = panelplus.utils.multiWayCluster(XFull, res, {id, year});
seMW = sqrt(diag(varcoefMW(1:4, 1:4)));

fprintf('\nTwo-way clustered SE (id x time):\n');
for j = 1:4
    fprintf('  %s: %.4f\n', fe.xnames{j}, seMW(j));
end

%% 2. Panel FGLS
fprintf('\n=== 2. Panel FGLS Demo ===\n');

% FGLS with AR(1) errors
estFGLS = panelplus.estimation.panelFGLS(id, year, y, X, ...
    'arType', 'ar1', 'panels', 'hetero', 'method', 'fe');
panelplus.utils.estdisp(estFGLS);

fprintf('AR(1) coefficient: %.4f\n', estFGLS.methodDetails.rho);

%% 3. Panel GMM (Simulated dynamic model)
fprintf('\n=== 3. Panel GMM Demo ===\n');

% Create lagged dependent variable for dynamic panel
yLag = NaN(size(y));
uid = unique(id);
for i = 1:length(uid)
    idx = find(id == uid(i));
    yLag(idx(2:end)) = y(idx(1:end-1));
end

% Remove first observation per individual (has NaN lag)
validIdx = ~isnan(yLag);
yValid = y(validIdx);
yLagValid = yLag(validIdx);
XValid = X(validIdx, :);
idValid = id(validIdx);
yearValid = year(validIdx);

% GMM estimation (simplified call)
try
    estGMM = panelplus.estimation.panelGMM(idValid, yearValid, yValid, ...
        [yLagValid, XValid], 'method', 'difference', 'lags', 1);
    panelplus.utils.estdisp(estGMM);

    fprintf('Hansen J-test: stat=%.2f, p=%.4f\n', ...
        estGMM.diagnostics.tests{1}.stat, estGMM.diagnostics.tests{1}.pval);
catch ME
    fprintf('GMM requires sufficient time periods. Error: %s\n', ME.message);
end

%% 4. Difference-in-Differences
fprintf('\n=== 4. DID Demo ===\n');

% Simulate treatment: first 20 states treated after 1980
treat = double(id <= 20);
post = double(year >= 1980);

estDID = panelplus.causal.did(id, year, y, X(:,1:2), treat, post, ...
    'method', 'twfe', 'cluster', id);
panelplus.utils.estdisp(estDID);

fprintf('ATT estimate: %.4f (SE: %.4f)\n', ...
    estDID.methodDetails.ATT, estDID.methodDetails.ATT_se);

%% 5. Staggered DID
fprintf('\n=== 5. Staggered DID Demo ===\n');

% Simulate staggered adoption
treatTime = NaN(length(y), 1);
for i = 1:length(uid)
    idx = (id == uid(i));
    if uid(i) <= 10
        treatTime(idx) = 1978;  % Early adopters
    elseif uid(i) <= 25
        treatTime(idx) = 1982;  % Late adopters
    else
        treatTime(idx) = Inf;   % Never treated
    end
end

try
    estStag = panelplus.causal.staggeredDID(id, year, y, X(:,1:2), treatTime, ...
        'method', 'cs', 'control', 'nevertreated', 'aggregation', 'simple');
    panelplus.utils.estdisp(estStag);

    fprintf('Aggregated ATT: %.4f\n', estStag.coef);
catch ME
    fprintf('Staggered DID error: %s\n', ME.message);
end

%% 6. Event Study
fprintf('\n=== 6. Event Study Demo ===\n');

try
    estES = panelplus.causal.eventStudy(id, year, y, X(:,1:2), treatTime, ...
        'leads', 4, 'lags', 6, 'reference', -1);

    fprintf('Event study coefficients:\n');
    for j = 1:length(estES.coef)
        fprintf('  %s: %.4f (%.4f)\n', estES.xnames{j}, estES.coef(j), estES.stderr(j));
    end

    fprintf('Pre-trend F-test: F=%.2f, p=%.4f\n', ...
        estES.methodDetails.preTestF.stat, estES.methodDetails.preTestF.pval);

    % Plot event study
    figure;
    panelplus.causal.eventStudyPlot(estES, 'title', 'Event Study: State Policy Effect');
catch ME
    fprintf('Event study error: %s\n', ME.message);
end

%% 7. RDD Demo
fprintf('\n=== 7. RDD Demo ===\n');

% Generate RDD data
rng(42);
runVar = randn(500, 1) * 2;
cutoff = 0;
treatment = double(runVar >= cutoff);
outcomeRDD = 0.5 * runVar + 3.0 * treatment + randn(500, 1) * 0.5;

estRDD = panelplus.causal.rdd(runVar, outcomeRDD, cutoff, ...
    'bwmethod', 'ik', 'kernel', 'triangular', 'order', 1);
panelplus.utils.estdisp(estRDD);

fprintf('RDD estimate: %.4f (SE: %.4f)\n', estRDD.coef, estRDD.stderr);
fprintf('Bandwidth: %.4f\n', estRDD.methodDetails.bandwidth);
fprintf('Effective N: left=%d, right=%d\n', ...
    round(estRDD.methodDetails.effNleft), round(estRDD.methodDetails.effNright));

%% 8. Panel VAR
fprintf('\n=== 8. Panel VAR Demo ===\n');

% Create multivariate panel data
Y_var = [log(gsp), log(pcap), log(emp)];

try
    estVAR = panelplus.timeseries.panelVAR(id, year, Y_var, 'lags', 2, 'method', 'ols');

    fprintf('VAR(%d) estimated\n', estVAR.methodDetails.nLags);
    fprintf('Stability: %s (max eigenvalue = %.4f)\n', ...
        string(estVAR.methodDetails.isStable), estVAR.methodDetails.maxEigenvalue);
    fprintf('AIC: %.2f, BIC: %.2f\n', estVAR.AIC, estVAR.BIC);

    % Impulse Response Functions
    irf = panelplus.timeseries.panelIRF(estVAR, 'horizon', 10, 'type', 'orthogonal', 'ci', false);

    fprintf('\nIRF of Var1 shock on Var1 (first 5 periods):\n');
    for h = 1:5
        fprintf('  h=%d: %.4f\n', h-1, irf.point(h, 1, 1));
    end

    % Granger causality test
    granger = panelplus.timeseries.panelGranger(estVAR, 2, 1);
    fprintf('\nGranger test (Var2 -> Var1): stat=%.2f, p=%.4f\n', ...
        granger.stat, granger.pval);
catch ME
    fprintf('Panel VAR error: %s\n', ME.message);
end

%% 9. DDML
fprintf('\n=== 9. Double/Debiased ML Demo ===\n');

% Create treatment variable
D_ml = treat .* post;

try
    estDDML = panelplus.ml.panelDDML(id, year, y, D_ml, X, ...
        'method', 'lasso', 'nFolds', 5, 'panelFE', true);
    panelplus.utils.estdisp(estDDML);

    fprintf('DDML ATE: %.4f (SE: %.4f)\n', estDDML.coef, estDDML.stderr);
    fprintf('R² for Y model: %.4f\n', estDDML.methodDetails.r2_Y);
    fprintf('R² for D model: %.4f\n', estDDML.methodDetails.r2_D);
catch ME
    fprintf('DDML error: %s\n', ME.message);
end

%% 10. Causal Forest
fprintf('\n=== 10. Causal Forest Demo ===\n');

try
    estCF = panelplus.ml.causalForest(X, y, D_ml, ...
        'nTrees', 100, 'honesty', true);
    panelplus.utils.estdisp(estCF);

    fprintf('ATE estimate: %.4f (SE: %.4f)\n', estCF.coef, estCF.stderr);
    fprintf('\nVariable importance:\n');
    for j = 1:min(4, length(estCF.methodDetails.variableImportance))
        fprintf('  x%d: %.4f\n', j, estCF.methodDetails.variableImportance(j));
    end

    % CATE heterogeneity
    cate = estCF.methodDetails.cate;
    fprintf('\nCATE summary:\n');
    fprintf('  Mean: %.4f, SD: %.4f\n', mean(cate), std(cate));
    fprintf('  Min: %.4f, Max: %.4f\n', min(cate), max(cate));
catch ME
    fprintf('Causal Forest error: %s\n', ME.message);
end

%% Summary
fprintf('\n');
fprintf('='.^(1:60));
fprintf('\n');
fprintf('PanelPlus Demo Complete!\n');
fprintf('='.^(1:60));
fprintf('\n');
fprintf('\nAvailable methods:\n');
fprintf('  panelplus.estimation: panelFGLS, panelGMM\n');
fprintf('  panelplus.causal: did, staggeredDID, eventStudy, rdd\n');
fprintf('  panelplus.timeseries: panelVAR, panelIRF, panelGranger\n');
fprintf('  panelplus.ml: panelDDML, causalForest\n');
fprintf('  panelplus.utils: multiWayCluster, panelResult, estdisp\n');
