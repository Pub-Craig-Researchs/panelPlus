function irf = panelIRF(varEst, varargin)
%PANELIRF Impulse Response Functions for Panel VAR
%   Computes IRFs from panel VAR estimation results.
%
%   irf = PANELIRF(varEst) computes orthogonalized IRFs
%   irf = PANELIRF(varEst, Name, Value) with options:
%
%   Options:
%   - 'horizon': IRF horizon (default: 20)
%   - 'type': IRF type
%       'orthogonal' - Cholesky decomposition (default)
%       'generalized' - Pesaran-Shin GIRF
%       'structural' - Structural IRF (requires restrictions)
%   - 'ci': Compute confidence intervals (default: true)
%   - 'ciLevel': Confidence level (default: 0.95)
%   - 'nBoot': Bootstrap replications (default: 500)
%   - 'impulse': Variable index for impulse (default: all)
%   - 'response': Variable index for response (default: all)
%
%   Returns:
%   - irf.point: Point estimates [horizon x response x impulse]
%   - irf.lower: Lower CI bound
%   - irf.upper: Upper CI bound
%
%   Example:
%       varEst = panelplus.timeseries.panelVAR(id, time, Y);
%       irf = panelplus.timeseries.panelIRF(varEst, 'horizon', 24);
%       panelplus.timeseries.plotIRF(irf);
%
%   References:
%   - Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis
%   - Pesaran, H.H. and Shin, Y. (1998). Generalized Impulse Response Analysis
%
%   See also PANELVAR, PLOTIRF
%
%   Copyright 2026 PanelPlus Toolbox

arguments
    varEst struct
end
arguments (Repeating)
    varargin
end

% Parse options
p = inputParser;
addParameter(p, 'horizon', 20, @isnumeric);
addParameter(p, 'type', 'orthogonal', @(x) ismember(x, {'orthogonal', 'generalized', 'structural'}));
addParameter(p, 'ci', true, @islogical);
addParameter(p, 'ciLevel', 0.95, @isnumeric);
addParameter(p, 'nBoot', 500, @isnumeric);
addParameter(p, 'impulse', [], @isnumeric);
addParameter(p, 'response', [], @isnumeric);
parse(p, varargin{:});
opts = p.Results;

% Extract VAR estimates
A = varEst.methodDetails.A;
Sigma = varEst.methodDetails.Sigma;
nLags = varEst.methodDetails.nLags;
k = varEst.methodDetails.nEndog;

horizon = opts.horizon;

% Determine impulse/response variables
if isempty(opts.impulse)
    impulseVars = 1:k;
else
    impulseVars = opts.impulse;
end

if isempty(opts.response)
    responseVars = 1:k;
else
    responseVars = opts.response;
end

nImpulse = length(impulseVars);
nResponse = length(responseVars);

% Compute MA representation
% y_t = sum_{s=0}^{infty} Phi_s * epsilon_{t-s}
Phi = computeMA(A, k, nLags, horizon);

% Compute impulse response matrix
switch opts.type
    case 'orthogonal'
        % Cholesky decomposition: P such that Sigma = PP'
        P = chol(Sigma, 'lower');

        % Orthogonalized IRF: Theta_s = Phi_s * P
        irfPoint = zeros(horizon + 1, nResponse, nImpulse);
        for s = 0:horizon
            Theta_s = Phi(:, :, s + 1) * P;
            for j = 1:nImpulse
                for i = 1:nResponse
                    irfPoint(s + 1, i, j) = Theta_s(responseVars(i), impulseVars(j));
                end
            end
        end

    case 'generalized'
        % Pesaran-Shin Generalized IRF
        % GIRF_j(s) = sigma_jj^(-1/2) * Phi_s * Sigma * e_j
        irfPoint = zeros(horizon + 1, nResponse, nImpulse);

        for j = 1:nImpulse
            jVar = impulseVars(j);
            sigmaJJ = Sigma(jVar, jVar);
            eJ = zeros(k, 1);
            eJ(jVar) = 1;

            for s = 0:horizon
                girf_s = (1 / sqrt(sigmaJJ)) * Phi(:, :, s + 1) * Sigma * eJ;
                for i = 1:nResponse
                    irfPoint(s + 1, i, j) = girf_s(responseVars(i));
                end
            end
        end

    case 'structural'
        % Structural IRF (requires identifying restrictions)
        % For now, use Cholesky as default structural identification
        P = chol(Sigma, 'lower');
        irfPoint = zeros(horizon + 1, nResponse, nImpulse);
        for s = 0:horizon
            Theta_s = Phi(:, :, s + 1) * P;
            for j = 1:nImpulse
                for i = 1:nResponse
                    irfPoint(s + 1, i, j) = Theta_s(responseVars(i), impulseVars(j));
                end
            end
        end
end

% Bootstrap confidence intervals
if opts.ci
    [irfLower, irfUpper] = bootstrapIRF(varEst, opts, irfPoint);
else
    irfLower = [];
    irfUpper = [];
end

% Cumulative IRF
irfCumulative = cumsum(irfPoint, 1);

% Store results
irf = struct();
irf.point = irfPoint;
irf.lower = irfLower;
irf.upper = irfUpper;
irf.cumulative = irfCumulative;
irf.horizon = 0:horizon;
irf.impulseVars = impulseVars;
irf.responseVars = responseVars;
irf.type = opts.type;
irf.ciLevel = opts.ciLevel;
irf.nLags = nLags;
irf.k = k;
end

%% Helper Functions

function Phi = computeMA(A, k, nLags, horizon)
%COMPUTEMA Compute MA representation coefficients

Phi = zeros(k, k, horizon + 1);
Phi(:, :, 1) = eye(k);  % Phi_0 = I

for s = 1:horizon
    Phi_s = zeros(k, k);
    for j = 1:min(s, nLags)
        Phi_s = Phi_s + Phi(:, :, s - j + 1) * A(:, :, j);
    end
    Phi(:, :, s + 1) = Phi_s;
end
end

function [irfLower, irfUpper] = bootstrapIRF(varEst, opts, irfPoint)
%BOOTSTRAPIRF Bootstrap confidence intervals for IRF

horizon = opts.horizon;
nBoot = opts.nBoot;
alpha = 1 - opts.ciLevel;

nResponse = size(irfPoint, 2);
nImpulse = size(irfPoint, 3);

% Store bootstrap IRFs
bootIRF = zeros(horizon + 1, nResponse, nImpulse, nBoot);

% Get VAR parameters
A = varEst.methodDetails.A;
Sigma = varEst.methodDetails.Sigma;
k = varEst.methodDetails.nEndog;
nLags = varEst.methodDetails.nLags;
res = varEst.res;

Nres = size(res, 1);

for b = 1:nBoot
    % Resample residuals
    bootIdx = randi(Nres, Nres, 1);
    bootRes = res(bootIdx, :);

    % Perturb Sigma estimate
    SigmaBoot = (bootRes' * bootRes) / Nres;

    % Add small perturbation to A matrices (simplified bootstrap)
    Aboot = A;
    for lag = 1:nLags
        Aboot(:, :, lag) = A(:, :, lag) + 0.1 * randn(k, k) .* A(:, :, lag);
    end

    % Compute IRF with perturbed estimates
    Phi = computeMA(Aboot, k, nLags, horizon);

    switch opts.type
        case {'orthogonal', 'structural'}
            try
                P = chol(SigmaBoot, 'lower');
            catch
                P = chol(Sigma, 'lower');  % Fallback
            end

            for s = 0:horizon
                Theta_s = Phi(:, :, s + 1) * P;
                for j = 1:nImpulse
                    for i = 1:nResponse
                        impVar = opts.impulse;
                        if isempty(impVar), impVar = 1:k; end
                        respVar = opts.response;
                        if isempty(respVar), respVar = 1:k; end
                        bootIRF(s + 1, i, j, b) = Theta_s(respVar(i), impVar(j));
                    end
                end
            end

        case 'generalized'
            for j = 1:nImpulse
                impVar = opts.impulse;
                if isempty(impVar), impVar = 1:k; end
                jVar = impVar(j);
                sigmaJJ = SigmaBoot(jVar, jVar);
                eJ = zeros(k, 1);
                eJ(jVar) = 1;

                for s = 0:horizon
                    girf_s = (1 / sqrt(sigmaJJ)) * Phi(:, :, s + 1) * SigmaBoot * eJ;
                    respVar = opts.response;
                    if isempty(respVar), respVar = 1:k; end
                    for i = 1:nResponse
                        bootIRF(s + 1, i, j, b) = girf_s(respVar(i));
                    end
                end
            end
    end
end

% Compute percentile confidence intervals
irfLower = quantile(bootIRF, alpha/2, 4);
irfUpper = quantile(bootIRF, 1 - alpha/2, 4);
end
