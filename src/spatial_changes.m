function [Iout, Achan, cllen, belt] = spatial_changes(Icp, Icl, Ianalyze, spacing, W, es, plotornot)

% Given an input image of migrated area over some elapsed time, this
% function generates a polygon encompassing the meander belt, then divides
% this polygon into equally-spaced smaller polygon segments based on a 
% given input spacing. Average migration rate is computed for each segment
% along the meander belt centerline.

% Requires the Image Processing toolbox

% INPUTS:        Icp - T-element cell, where each element contains the
%                      binary image of the channel mask area. T is the
%                      number of realizations (years). This is used to
%                      generate the channel belt boundaries.
%                Icl - T-element cell, where each element contains the
%                      binary image of the channel centerline.
%           Ianalyze - a Wx1 cell, where W is the number of input area
%                      types to be analyzed (e.g. centerline migration, 
%                      accretion, erosion, etc.). Each entry of the cell is
%                      itself a Q element cell, where Q is the number of
%                      realizations (or years) of each area image. The cell
%                      contains binary images of the areas to be analyzed.
%                 es - exit sides of channel, or sides of the image which
%                      intersect the channel mask. Upstream side is first.
%            spacing - distance (in pixels) along the meander valley 
%                      centerline at which to average migration rates
%                  W - nominal channel width
%
% OUTPUTS:      Iout - a Wx1 cell, where each element corresponds to the
%                      inputs in Ianalyze. Each element contains an SxQ
%                      matrix where each element contains the number of
%                      pixels within each Sth buffer polygon. S is the 
%                      number of buffer polygons and Q is the number of 
%                      realizations (years) of input imagery.
%              Achan - SxQ matrix where each element contains the number of
%                      channel mask pixels in each buffer polygon of each
%                      realization (or year).
%              cllen - SxQ matrix where each element contains the length of
%                      centerline within each buffer polygon of each
%                      realization (or year).
%               belt - structure containing the meander belt centerline,
%                      left, and right edge coordinates

Nims = numel(Icp);
Na = numel(Ianalyze);

%% First we construct a polygon buffer that encompasses all the migrated area.

% Must pad the buffer so it has "room to grow" when being dilated; pad
% all images initially so image sizes are consistent
imsizeo = size(Icp{1});
padadd = W*20; % How much padding to add
% Pad all migration images
for i = 1:Nims
    Icp{i} = padarray(Icp{i},[padadd padadd]);
end
impadded = size(Icp{1});

% Make an image of all channel positions 
Icpall = false(impadded);
for i = 1:numel(Icp)
    Icpall(Icp{i}) = true;
end

% Generate banklines of meander belt and oversmooth them, then check that 
% they contain all the migrated areas. If not, repeat the process with more
% dilation.
Ibuffer = Icpall;
ndilate = round(W*10); % number of initial dilations
ct = 0; % for counting number of dilations required (diagnostic)
while 1
    Ibuffer = bwmorph(Ibuffer,'dilate',ndilate);
    banksout = banklines_from_mask(Ibuffer,es,0);
    lb = banksout{1};
    rb = banksout{2};
    lbs = savfilt(lb,W*50);
    rbs = savfilt(rb,W*50);
    polymaskx = [lbs(:,1); flipud(rbs(:,1)); lbs(1,1)];
    polymasky = [lbs(:,2); flipud(rbs(:,2)); lbs(1,2)];
    Ibuffer = poly2mask(polymaskx, polymasky, impadded(1), impadded(2)); 
    ct = ct + 1;
    
    % Check that mask covers all migrated area, increase dilation if not
    if sum(sum(Ibuffer & Icpall)) ~= sum(sum(Icpall))
        ndilate = round(ndilate + W/3);
    else
        break
    end
end

%% Now that meander belt extents are known, divide them into buffer polygons

% Clip the banklines to the centerline extents
for i = 1:2
    if strcmp(es(i),'N')
        lbs(lbs(:,2) <= padadd,:) = [];
        rbs(rbs(:,2) <= padadd,:) = [];
    elseif strcmp(es(i),'S')
        lbs(lbs(:,2) > imsizeo(1)+padadd,:) = [];
        rbs(rbs(:,2) > imsizeo(1)+padadd,:) = [];
    elseif strcmp(es(i),'W')
        lbs(lbs(:,1) <= padadd,:) = [];
        rbs(rbs(:,1) <= padadd,:) = [];
    elseif strcmp(es(i),'E')
        lbs(lbs(:,1) > imsizeo(2)+padadd,:) = [];
        rbs(rbs(:,1) > imsizeo(2)+padadd,:) = [];
    end
end

% Fit the edgelines with splines of only a few points for smoother curvatures
% First, coarsely sample the edgelines
len_lb = sum(sqrt(diff(lbs(:,1)).^2+diff(lbs(:,2)).^2));
len_rb = sum(sqrt(diff(rbs(:,1)).^2+diff(rbs(:,2)).^2));
avglen = (len_lb+len_rb)/2;
npts = round(avglen/(W*20));
tlb = linspace(0,1,npts);
trb = linspace(0,1,npts);
lbcoarse = interparc(tlb,lbs(:,1),lbs(:,2),'linear');
rbcoarse = interparc(trb,rbs(:,1),rbs(:,2),'linear');

% Next, resample the coarse edgelines using interpolating piecewise splines
[~,~,lbspline] = interparc(tlb,lbcoarse(:,1),lbcoarse(:,2),'spline');
[~,~,rbspline] = interparc(trb,rbcoarse(:,1),rbcoarse(:,2),'spline');

% Evaluate the coarse splines at a finer density
nints = round(avglen/(W/10));
t = linspace(0,1,nints);
lbrs = lbspline(t);
rbrs = rbspline(t);

% Generate a centerline as the midpoint of the left and right edgelines
clrs = [(lbrs(:,1)+rbrs(:,1))/2 (lbrs(:,2)+rbrs(:,2))/2];

% Find the centerline curvature
Ccl = curvatures(clrs);

% First two curvatures are NaN; reset them
Ccl(1:2) = mean(Ccl(isnan(Ccl)==0));

% Find inflection points of the centerline
ipcl = find(abs(diff(sign(Ccl)))==2);

% Pre-process centerline inflection points
% Distances along centerline
scl = [0; cumsum(sqrt(diff(clrs(:,1)).^2+diff(clrs(:,2)).^2))];

% Remove inflection points too close to ends of centerline
threshdist = spacing*3; % remove inflection points that are less than this distance from end of centerline
us_dists = scl(ipcl);
scl_ds = scl(end)-scl;
ds_dists = scl_ds(ipcl);
remove = find(us_dists < threshdist | ds_dists < threshdist);
ipcl(remove) = [];

% Combine inflection points that are too close to each other
threshdist2 = spacing*3;
while 1
    dists = scl(ipcl(2:end))-scl(ipcl(1:end-1));
    tooclose = find(dists<threshdist2,1,'first');
    if isempty(tooclose)
        break
    end
    middist = scl(ipcl(tooclose)) + (scl(ipcl(tooclose+1))-scl(ipcl(tooclose)))/2;
    [~,newidx] = min(abs(scl-middist));
    ipcl(tooclose) = newidx;
    ipcl(tooclose+1) = [];
end

% Find left and right edgeline points corresponding to each centerline 
% inflection point by the intersection of perpendiculars to the centerline 
% and each edgline

% Uncomment for plotting (diagnostic)
% close all
% imshow(Imigall); hold on
% plot(clrs(:,1),clrs(:,2),'m')
% plot(clrs(ipcl,1),clrs(ipcl,2),'co')
% plot(lbrs(:,1),lbrs(:,2),'r');
% plot(rbrs(:,1),rbrs(:,2),'r');

dx = sum(sum(Icpall))/avglen*4; % Length of perpendicular lines
% Initialize storage
intlb = zeros(numel(ipcl),1);
intrb = intlb;
for a = 1:numel(ipcl)
    tpx = clrs(ipcl(a),1);
    tpy = clrs(ipcl(a),2);
    
    % Slope of line from current centerline point to point on circle
    m = (tpy-clrs(ipcl(a)-1,2))/(tpx-clrs(ipcl(a)-1,1));
    
    % Slope of perpendicular line
    minv = -1/m;
    
    % Construct perpendicular lines
    upper_pt = [tpx + dx, tpy + dx*minv];
    lower_pt = [tpx - dx, tpy - dx*minv];
    perpx = [upper_pt(1) lower_pt(1)];
    perpy = [upper_pt(2) lower_pt(2)];
    
%     close all
%     plot(lbrs(:,1),-lbrs(:,2)); axis equal; hold on
%     plot(rbrs(:,1),-rbrs(:,2)); 
%     plot(perpx,-perpy);
    
    % Find intersections
    [~,~,~,lbidx] = intersections(perpx,perpy,lbrs(:,1),lbrs(:,2),0);
    [~,~,~,rbidx] = intersections(perpx,perpy,rbrs(:,1),rbrs(:,2),0);

    intlb(a) = round(lbidx);
    intrb(a) = round(rbidx);
    
%     plot(perpx,perpy,'m')
end
% plot(lbrs(intlb,1),lbrs(intlb,2),'ro')
% plot(rbrs(intrb,1),rbrs(intrb,2),'ro')

% Distances between nodes of edgelines
slb = [0; cumsum(sqrt(diff(lbrs(:,1)).^2+diff(lbrs(:,2)).^2))];
srb = [0; cumsum(sqrt(diff(rbrs(:,1)).^2+diff(rbrs(:,2)).^2))];

% Compute the number of buffers to create between centerline inflection points
nbreaks = nan(numel(ipcl),1);
for i = 1:numel(ipcl)+1
    if i == 1
        stidx = 1;
    else
        stidx = ipcl(i-1);
    end
    if i == numel(ipcl)+1
        enidx = length(clrs);
    else
        enidx = ipcl(i);
    end
    nbreaks(i) = floor((scl(enidx)-scl(stidx))/spacing);
end

% Re-parameterize edgelines based on nbreaks
tlb = [];
for i = 1:numel(nbreaks)
    if i == 1
        stidx = 1;
    else
        stidx = intlb(i-1);
    end
    if i == numel(intlb)+1
        enidx = length(lbrs);
    else
        enidx = intlb(i);
    end
    tlb = [tlb linspace(slb(stidx),slb(enidx),nbreaks(i))/slb(end)];
end

trb = [];
for i = 1:numel(nbreaks)
    if i == 1
        stidx = 1;
    else
        stidx = intrb(i-1);
    end
    if i == numel(intrb)+1
        enidx = length(rbrs);
    else
        enidx = intrb(i);
    end
    trb = [trb linspace(srb(stidx),srb(enidx),nbreaks(i))/srb(end)];
end

% Ensure endpoints are included in parameterization
tlb = unique([0 tlb 1]);
trb = unique([0 trb 1]);

% Resample to form buffer polygons
lbrs = lbspline(tlb);
rbrs = rbspline(trb);

% Plot to ensure buffer polygons are properly constructed
if plotornot
    figure
    imshow(Icpall); hold on
    plot(lbrs(:,1),lbrs(:,2),'m')
    plot(rbrs(:,1),rbrs(:,2),'m')
    for i = 1:length(lbrs)
        plot([lbrs(i,1) rbrs(i,1)],[lbrs(i,2) rbrs(i,2)],'m');
    end
    pause(.1)
end

% Recompute centerline to match resampled edgelines
clrs = [(lbrs(:,1)+rbrs(:,1))/2 (lbrs(:,2)+rbrs(:,2))/2];

% Distance along centerline
scl = [0; cumsum(sqrt(diff(clrs(:,1)).^2+diff(clrs(:,2)).^2))];

% Midpoint distance of each segment
Smid = (scl(1:end-1)+scl(2:end))/2;

%% Compute actual centerline lengths for all Icl images
for i = 1:Nims
    I = padarray(Icl{i},[padadd padadd]);
    [Ey, Ex] = find(bwmorph(I,'endpoints')); 
    clxy{i} = skeleton_coords(I, Ex(1), Ey(1), es);
    S{i} = [0; cumsum(sqrt(diff(clxy{i}(:,1)).^2+diff(clxy{i}(:,2)).^2))];
end

%% Compute migrated area, channel area, and centerline length in each buffer for each year
% Initialize storage
NIa = numel(Ianalyze{1});
Achan = nan(length(rbrs)-1,NIa);
cllen = Achan;
for j = 1:Na
    Iout{j} = Achan;
end

% Apply buffer polygons to each input image set
t1 = tic;
for i = 1:length(rbrs)-1 % For each buffer polygon
    
 
    % Make a polygon using resampled buffer edges and centerline
    polyclipx = [lbrs(i,1); rbrs(i,1); rbrs(i+1,1); lbrs(i+1,1)];
    polyclipy = [lbrs(i,2); rbrs(i,2); rbrs(i+1,2); lbrs(i+1,2)];

    % Create buffer raster from the polygon
    Ibuffer = poly2mask(polyclipx,polyclipy, impadded(1), impadded(2));
    % There is no overlap between Ibuffer(i), Ibuffer(i-1), or Ibuffer(i+1)
    
    % Initialize storage
    migs = nan(NIa,1);
    careas = migs;
    lens = zeros(NIa,1);
    % For each realization (year)
    for ii = 1:NIa  
        
        % Centerline length for each year (river centerline, not buffer
        % centerline)
        Iclbuf = padarray(Icl{ii},[padadd padadd]) & Ibuffer;
        rp = regionprops(Iclbuf,'PixelIdxList');
        for iii = 1:numel(rp)
            Ilen = false(size(Iclbuf));
            Ilen(rp(iii).PixelIdxList) = true;
            [Ey, Ex] = find(bwmorph(Ilen,'endpoints')); 
            if numel(Ey) == 1 % where there is only a single pixel
                lens(ii) = lens(ii) + 1;
                continue
            end
            [~,idx1] = min(sqrt((Ex(1)-clxy{ii}(:,1)).^2+(Ey(1)-clxy{ii}(:,2)).^2));
            [~,idx2] = min(sqrt((Ex(2)-clxy{ii}(:,1)).^2+(Ey(2)-clxy{ii}(:,2)).^2));
            lens(ii) = lens(ii) + abs(S{ii}(idx1)-S{ii}(idx2));
        end
            
        % Channel mask area for each year
        careas(ii) = sum(sum(Ibuffer & Icp{ii}));
        
        % Input images areas for each year
        for j = 1:Na
            areas(ii,j) = sum(sum(Ibuffer & padarray(Ianalyze{j}{ii}, [padadd padadd])));
        end
    end
    
    % Store results
    Achan(i,:) = careas;
    cllen(i,:) = lens;
    for j = 1:Na
        Iout{j}(i,:) = areas(:,j);
    end
    
    t2 = toc(t1);
    t_est = (t2)/i;
    disp(['Spatial change: buffer ',num2str(i),'/',num2str(length(rbrs)-1),' is complete. Approximately ',num2str(((length(rbrs)-1)-i)*(t_est)/60),' minutes remaining.']);
    tic
end


% Remove padding from buffer coordinates
belt.lb = lbrs - padadd;
belt.rb = rbrs - padadd;
belt.cl = clrs - padadd;
belt.Smid = Smid;

