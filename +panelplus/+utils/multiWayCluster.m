function varcoef = multiWayCluster(X, res, clusterVars, varargin)
%MULTIWAYCLUSTER Multi-way clustered variance-covariance matrix
%   Computes multi-way clustered standard errors using the 
%   Cameron-Gelbach-Miller (2011) intersection clustering method.
%
%   varcoef = MULTIWAYCLUSTER(X, res, clusterVars) computes the clustered
%   variance-covariance matrix for regression with design matrix X, 
%   residuals res, and cluster variables clusterVars (cell array).
%
%   varcoef = MULTIWAYCLUSTER(X, res, clusterVars, Name, Value) with options:
%   - 'dfCorrection': Apply small-sample correction (default: true)
%
%   The method implements:
%   V_multiway = V_c1 + V_c2 + ... - V_c1∩c2 - V_c1∩c3 - ... + V_c1∩c2∩c3 + ...
%
%   Reference:
%   Cameron, A.C., Gelbach, J.B., Miller, D.L. (2011). Robust Inference 
%   With Multiway Clustering. Journal of Business & Economic Statistics.
%
%   See also PANEL, OLS
%
%   Copyright 2026 PanelPlus Toolbox

    arguments
        X (:,:) double
        res (:,1) double
        clusterVars cell
    end
    arguments (Repeating)
        varargin
    end
    
    % Parse options
    p = inputParser;
    addParameter(p, 'dfCorrection', true, @islogical);
    parse(p, varargin{:});
    opts = p.Results;
    
    N = size(X, 1);
    k = size(X, 2);
    numClusterDims = length(clusterVars);
    
    % Validate cluster variables
    for i = 1:numClusterDims
        if length(clusterVars{i}) ~= N
            error('Cluster variable %d has different length than data', i);
        end
    end
    
    % Compute (X'X)^(-1)
    invXX = (X' * X) \ eye(k);
    
    % Generate all possible cluster combinations (inclusion-exclusion)
    % For 2 clusters: {1}, {2}, {1,2}
    % For 3 clusters: {1}, {2}, {3}, {1,2}, {1,3}, {2,3}, {1,2,3}
    allCombinations = cell(2^numClusterDims - 1, 1);
    idx = 1;
    for r = 1:numClusterDims
        combos = nchoosek(1:numClusterDims, r);
        for j = 1:size(combos, 1)
            allCombinations{idx} = combos(j, :);
            idx = idx + 1;
        end
    end
    
    % Initialize variance-covariance matrix
    varcoef = zeros(k, k);
    
    % Apply inclusion-exclusion principle
    for c = 1:length(allCombinations)
        combo = allCombinations{c};
        numInCombo = length(combo);
        
        % Create intersection cluster ID
        intersectCluster = createIntersectionCluster(clusterVars, combo);
        
        % Compute clustered "meat" for this combination
        [meatMatrix, nClusters] = computeClusterMeat(X, res, intersectCluster);
        
        % Compute variance for this cluster combination
        V_combo = invXX * meatMatrix * invXX;
        
        % Apply small-sample correction
        if opts.dfCorrection
            correction = nClusters / (nClusters - 1) * (N - 1) / (N - k);
            V_combo = correction * V_combo;
        end
        
        % Inclusion-exclusion: add if odd number of dimensions, subtract if even
        if mod(numInCombo, 2) == 1
            varcoef = varcoef + V_combo;
        else
            varcoef = varcoef - V_combo;
        end
    end
    
    % Ensure positive semi-definiteness via eigenvalue correction
    varcoef = ensurePSD(varcoef);
end

function intersectCluster = createIntersectionCluster(clusterVars, indices)
%CREATEINTERSECTIONCLUSTER Create intersection cluster IDs
    N = length(clusterVars{indices(1)});
    
    % Combine cluster variables into string representation
    clusterStrings = strings(N, 1);
    for i = 1:length(indices)
        cv = clusterVars{indices(i)};
        if isnumeric(cv)
            cvStr = string(cv);
        else
            cvStr = string(cv);
        end
        clusterStrings = clusterStrings + "_" + cvStr;
    end
    
    % Convert to numeric cluster IDs
    [~, ~, intersectCluster] = unique(clusterStrings);
end

function [meatMatrix, nClusters] = computeClusterMeat(X, res, clusterID)
%COMPUTECLUSTERMEAT Compute the "meat" matrix for cluster-robust variance
    uniqueClusters = unique(clusterID);
    nClusters = length(uniqueClusters);
    k = size(X, 2);
    
    meatMatrix = zeros(k, k);
    
    for g = 1:nClusters
        idx = (clusterID == uniqueClusters(g));
        Xg = X(idx, :);
        resG = res(idx);
        
        % Sum of X'e for this cluster
        sumXe = Xg' * resG;
        
        % Outer product
        meatMatrix = meatMatrix + sumXe * sumXe';
    end
end

function A = ensurePSD(A)
%ENSUREPSD Ensure matrix is positive semi-definite
    % Symmetrize
    A = (A + A') / 2;
    
    % Eigenvalue correction
    [V, D] = eig(A);
    d = diag(D);
    
    % Set negative eigenvalues to small positive value
    minEig = max(d) * 1e-10;
    d(d < minEig) = minEig;
    
    A = V * diag(d) * V';
    A = (A + A') / 2;  % Re-symmetrize for numerical stability
end
