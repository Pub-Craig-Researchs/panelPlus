function estdisp(est)
%ESTDISP Display estimation results for PanelPlus
%   Displays formatted estimation output for panel results.
%
%   ESTDISP(est) displays the estimation results
%
%   See also PANELRESULT
%
%   Copyright 2026 PanelPlus Toolbox

% Header
fprintf('\n');
fprintf('================================================================================\n');
fprintf('%s Estimation\n', est.method);
fprintf('================================================================================\n');

% Sample info
fprintf('Observations:  %d\n', est.N);
if isfield(est, 'n') && est.n > 0
    fprintf('Groups (n):    %d\n', est.n);
end
if isfield(est, 'T') && est.T > 0
    fprintf('Time periods:  %d\n', est.T);
end

fprintf('\n');

% Coefficient table
k = length(est.coef);

% Get variable names
if isfield(est, 'xnames') && ~isempty(est.xnames)
    varNames = est.xnames;
else
    varNames = cell(k, 1);
    for j = 1:k
        varNames{j} = sprintf('x%d', j);
    end
end

% Print header
fprintf('%-15s %12s %12s %12s %12s\n', 'Variable', 'Coef.', 'Std.Err.', 't-stat', 'P>|t|');
fprintf('--------------------------------------------------------------------------------\n');

% Print coefficients
for j = 1:k
    if length(varNames) >= j
        name = varNames{j};
    else
        name = sprintf('x%d', j);
    end

    if length(name) > 15
        name = name(1:15);
    end

    coef = est.coef(j);
    if length(est.stderr) >= j
        se = est.stderr(j);
    else
        se = NaN;
    end

    if length(est.tstat) >= j
        t = est.tstat(j);
    else
        t = coef / se;
    end

    if length(est.pval) >= j
        pval = est.pval(j);
    else
        pval = 2 * (1 - normcdf(abs(t)));
    end

    % Significance stars
    if pval < 0.001
        stars = '***';
    elseif pval < 0.01
        stars = '**';
    elseif pval < 0.05
        stars = '*';
    elseif pval < 0.1
        stars = '.';
    else
        stars = '';
    end

    fprintf('%-15s %12.4f %12.4f %12.2f %10.4f %s\n', ...
        name, coef, se, t, pval, stars);
end

fprintf('--------------------------------------------------------------------------------\n');
fprintf('Signif. codes: 0 ''***'' 0.001 ''**'' 0.01 ''*'' 0.05 ''.'' 0.1\n');

% Goodness of fit
if isfield(est, 'r2') && ~isnan(est.r2)
    fprintf('\nR-squared:     %.4f\n', est.r2);
end
if isfield(est, 'adjr2') && ~isnan(est.adjr2)
    fprintf('Adj R-squared: %.4f\n', est.adjr2);
end

% Information criteria
if isfield(est, 'AIC') && ~isnan(est.AIC)
    fprintf('AIC:           %.2f\n', est.AIC);
end
if isfield(est, 'BIC') && ~isnan(est.BIC)
    fprintf('BIC:           %.2f\n', est.BIC);
end

% Diagnostic tests
if isfield(est, 'diagnostics') && isfield(est.diagnostics, 'tests') && ~isempty(est.diagnostics.tests)
    fprintf('\nDiagnostic Tests:\n');
    for i = 1:length(est.diagnostics.tests)
        test = est.diagnostics.tests{i};
        if isstruct(test)
            fprintf('  %s: stat = %.3f, p = %.4f\n', test.name, test.stat, test.pval);
        end
    end
end

fprintf('\n');
end
