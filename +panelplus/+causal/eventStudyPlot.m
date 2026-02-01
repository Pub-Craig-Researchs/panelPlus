function fig = eventStudyPlot(est, varargin)
%EVENTSTUDYPLOT Plot event study results
%   Creates a publication-quality event study plot with confidence intervals.
%
%   fig = EVENTSTUDYPLOT(est) plots the event study coefficients
%   fig = EVENTSTUDYPLOT(est, Name, Value) with options:
%
%   Options:
%   - 'ciLevel': Confidence level (default: 0.95)
%   - 'showCI': Show confidence intervals (default: true)
%   - 'showPreTest': Show pre-trend test results (default: true)
%   - 'title': Plot title (default: 'Event Study')
%   - 'xlabel': X-axis label (default: 'Time Relative to Treatment')
%   - 'ylabel': Y-axis label (default: 'Treatment Effect')
%   - 'refLine': Show reference line at 0 (default: true)
%   - 'markerSize': Marker size (default: 8)
%   - 'lineWidth': Line width (default: 1.5)
%   - 'color': Main color [r,g,b] (default: [0.2, 0.4, 0.8])
%
%   See also EVENTSTUDY
%
%   Copyright 2026 PanelPlus Toolbox

arguments
    est struct
end
arguments (Repeating)
    varargin
end

% Parse options
p = inputParser;
addParameter(p, 'ciLevel', 0.95, @isnumeric);
addParameter(p, 'showCI', true, @islogical);
addParameter(p, 'showPreTest', true, @islogical);
addParameter(p, 'title', 'Event Study', @ischar);
addParameter(p, 'xlabel', 'Time Relative to Treatment', @ischar);
addParameter(p, 'ylabel', 'Treatment Effect', @ischar);
addParameter(p, 'refLine', true, @islogical);
addParameter(p, 'markerSize', 8, @isnumeric);
addParameter(p, 'lineWidth', 1.5, @isnumeric);
addParameter(p, 'color', [0.2, 0.4, 0.8], @isnumeric);
parse(p, varargin{:});
opts = p.Results;

% Get data from estimation result
relTime = est.methodDetails.relTime;
coef = est.coef;
stderr = est.stderr;
ref = est.methodDetails.reference;

% Calculate confidence intervals
alpha = 1 - opts.ciLevel;
z = norminv(1 - alpha/2);
ciLo = coef - z * stderr;
ciHi = coef + z * stderr;

% Create figure
fig = figure('Position', [100, 100, 800, 500]);
hold on;

% Shade pre-treatment period
preIdx = relTime < 0;
if any(preIdx)
    xPre = [min(relTime(preIdx))-0.5, 0-0.5, 0-0.5, min(relTime(preIdx))-0.5];
    yLims = [min(ciLo)*1.5, max(ciHi)*1.5];
    if yLims(1) > 0, yLims(1) = -max(ciHi)*0.5; end
    if yLims(2) < 0, yLims(2) = abs(min(ciLo))*0.5; end
    yPre = [yLims(1), yLims(1), yLims(2), yLims(2)];
    fill(xPre, yPre, [0.9, 0.9, 0.95], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
end

% Reference line at y=0
if opts.refLine
    yline(0, 'k--', 'LineWidth', 0.8);
end

% Reference time line
xline(ref + 0.5, 'r:', 'LineWidth', 1, 'Label', 'Treatment');

% Confidence interval bands
if opts.showCI
    fill([relTime, fliplr(relTime)], [ciLo', fliplr(ciHi')], ...
        opts.color, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end

% Point estimates
plot(relTime, coef, 'o-', 'Color', opts.color, 'MarkerFaceColor', opts.color, ...
    'MarkerSize', opts.markerSize, 'LineWidth', opts.lineWidth);

% Reference period marker
refIdx = (relTime == ref);
plot(relTime(refIdx), coef(refIdx), 's', 'Color', 'k', 'MarkerFaceColor', 'none', ...
    'MarkerSize', opts.markerSize + 2, 'LineWidth', 2);

% Labels
xlabel(opts.xlabel, 'FontSize', 12);
ylabel(opts.ylabel, 'FontSize', 12);
title(opts.title, 'FontSize', 14);

% Grid
grid on;
box on;

% Set axis limits
xlim([min(relTime)-0.5, max(relTime)+0.5]);

% Pre-trend test annotation
if opts.showPreTest && isfield(est.methodDetails, 'preTestF')
    preTest = est.methodDetails.preTestF;
    if ~isnan(preTest.stat)
        annotStr = sprintf('Pre-trend F-test: %.2f (p=%.3f)', preTest.stat, preTest.pval);
        text(0.02, 0.98, annotStr, 'Units', 'normalized', ...
            'VerticalAlignment', 'top', 'FontSize', 10, ...
            'BackgroundColor', 'white', 'EdgeColor', 'black');
    end
end

hold off;
end
