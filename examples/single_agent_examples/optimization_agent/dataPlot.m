clear;
dataset = load("C:\Users\393258\Documents\Gitlab\ursa\examples\single_agent_examples\optimization_agent\dataMaterials.csv");
%%
comps = dataset(:,1:4);
yields = dataset(:,5);
elements = ["Ta","Nb","Mo","W"];
%%
figure(1);
for i=1:length(elements)
    subplot(2,2,i);
    scatter(comps(:,i), yields,50,yields,"filled");
    % Add a color bar to show the value-color mapping
    colorbar;
    % Optional: Customize the colormap (e.g., 'turbo', 'parula', 'jet')
    colormap turbo;
    title(["Element: "+string(elements(i))]);
    ylabel('Yield (GPa)');
    xlabel('Composition');
end
sgtitle("Yield vs Composition (per element)");

%%
thresh = 1.8;
idxthresh = find(yields>thresh);
bestYields = yields(idxthresh);
bestComps = comps(idxthresh,:);
%%
seriesNames = "Composition " + string(1:length(bestYields));

% (Optional) if you need a cell array of char vectors (older legend preferences)
seriesNamesCell = cellstr(seriesNames);

spiderPlot(bestComps,elements,seriesNamesCell,"Compositions with Yield>"+num2str(thresh),[0,1],5, bestYields);


function spiderPlot(data, labels,seriesNames, titleName, rLimits, nRings, mag)

    [nSeries, nAxes] = size(data);

    if isempty(labels)
        labels = compose("Var %d", 1:nAxes);
    else
        labels = string(labels);
        if numel(labels) ~= nAxes
            error("labels must have length %d (number of columns in data).", nAxes);
        end
    end

    if isempty(seriesNames)
        seriesNames = compose("Series %d", 1:nSeries);
    else
        seriesNames = string(seriesNames);
        if numel(seriesNames) ~= nSeries
            error("seriesNames must have length %d (number of rows in data).", nSeries);
        end
    end

    if any(isnan(rLimits))
        rMin = 0;
        rMax = max(data(:));
        if rMax == 0, rMax = 1; end
    else
        rMin = rLimits(1);
        rMax = rLimits(2);
        if rMax <= rMin
            error("rLimits must satisfy rMax > rMin.");
        end
    end

    % Angles (close the loop)
    theta = linspace(0, 2*pi, nAxes+1);
    theta(end) = theta(1);

    figure; hold on; axis equal; axis off;

    % Grid rings
    ringVals = linspace(rMin, rMax, nRings+1);
    ringVals = ringVals(2:end); % skip rMin
    for r = ringVals
        xg = (r-rMin)/(rMax-rMin) * cos(theta);
        yg = (r-rMin)/(rMax-rMin) * sin(theta);
        plot(xg, yg, 'k:', 'HandleVisibility','off');
    end

    % Spokes + labels
    for k = 1:nAxes
        plot([0 cos(theta(k))], [0 sin(theta(k))], 'k:', 'HandleVisibility','off');

        lx = 1.12*cos(theta(k));
        ly = 1.12*sin(theta(k));
        text(lx, ly, labels(k), ...
            'HorizontalAlignment','center', 'VerticalAlignment','middle');
    end

    mag = mag(:);                 % <-- you provide this (nSeries x 1)
    
    cmap = jet(256);
    clim = [min(mag) max(mag)];
    
    t = (mag - clim(1)) / (clim(2) - clim(1) + eps);
    t = max(0, min(1, t));
    idx = 1 + round(t * (size(cmap,1)-1));
    lineColors = cmap(idx, :); 

    % Plot series (normalize to [0,1] for drawing)
    for s = 1:nSeries
        r = data(s,:);
        rn = (r - rMin) / (rMax - rMin);         % normalize
        rn = max(0, min(1, rn));                 % clamp to [0,1]
        rn = [rn, rn(1)];                        % close loop

        xs = rn .* cos(theta);
        ys = rn .* sin(theta);

        h = plot(xs, ys, '-', 'LineWidth', 2);   % <-- get handle
        h.Color = lineColors(s,:);               % <-- set color from magnitude
        scatter(xs(1:end-1), ys(1:end-1), 50, 'o', 'filled', ...
            'HandleVisibility','off');
    end
    
    colormap(cmap);
    cb = colorbar;
    caxis(clim);
    cb.Label.String = "Magnitude";
    legend(seriesNames, 'Location','bestoutside');
    title(titleName);

    hold off;
end
