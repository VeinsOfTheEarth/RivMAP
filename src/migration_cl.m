function [Imig,Icutsall,cutidcs,cutarea,cutlen,chutelen] = migration_cl(xy1, xy2, es1, es2, W, sizeI)
% Computes an image of the area between two lines defined by x,y 
% coordinates. Then cutoffs are identified wherever the centerline length
% has been shortened more than 2W, and some cutoff properties are computed.

% Requires the Image Processing toolbox

% INPUTS:      xy1 - Nx2 array of centerline (or bankline) coordinates at
%                    t1. Can be smoothed or unsmoothed.
%              xy2 - Nx2 array of centerline (or bankline) coordinates at
%                    t2. Can be smoothed or unsmoothed.
%              es1 - sides of the image that xy1 intersects (e.g.
%                    'SN','EW') with upstream side listed first
%              es2 - sides of the image that xy2 intersects (e.g.
%                    'SN','EW') with upstream side listed first
%            sizeI - 2 element vector containing the number of rows and
%                    columns of the original image (that xy1,xy2 were
%                    derived from)
%
% OUTPUTS:    Imig - binary image of migrated area between xy1 and xy2 
%         Icutsall - binary image of cutoffs
%          cutidcs - Mx2 array of indices along xy1 where cutoffs occurred.
%                    The first index is the upstream cutoff node and the
%                    second is the downstream cutoff node. M is the number
%                    of cutoffs.
%          cutarea - Mx1 array containing areas (in pixels) of each cutoff.
%           cutlen - Mx1 array containing lengths (in pixels) of each
%                    cutoff.
%         chutelen - Mx1 array containing the length of the chute channel
%                    (the distance along xy2 between the two cutoff points)

% Round coordinates so they reference pixel indices
xy1 = round(xy1);
xy2 = round(xy2);

% In case the centerlines have different exit sides, choose the smaller
ar1 = max(xy1) - min(xy1);
ar2 = max(xy2) - min(xy2);
if prod(ar1) < prod(ar2)
    es = es1;
else
    es = es2;
end

% Initialize storage
Icrop1 = false(sizeI);
Icrop2 = false(sizeI);

yTop = []; xLeft = [];

% Loops to put xy1, xy2 in image format
for g = 1:length(xy1)
    Icrop1(xy1(g,2),xy1(g,1)) = true;
end
for g = 1:length(xy2)
    Icrop2(xy2(g,2),xy2(g,1)) = true;
end

% Crop images
if ismember('S',es) == 1
    yBottom = min(max(xy1(:,2)),max(xy2(:,2)));
    Icrop1(yBottom:end,:)=[];
    Icrop2(yBottom:end,:)=[];
end

if ismember('N',es) == 1
    yTop = max(min(xy1(:,2)), min(xy2(:,2)));
    Icrop1(1:yTop,:)=[];
    Icrop2(1:yTop,:)=[];
end

if ismember('E',es) == 1
    xRight = min(max(xy1(:,1)),max(xy2(:,1)));
    Icrop1(:,xRight:end) = [];
    Icrop2(:,xRight:end) = [];
end

if ismember('W',es) == 1
    xLeft = max(min(xy1(:,1)), min(xy2(:,1)));
    Icrop1(:,1:xLeft) = [];
    Icrop2(:,1:xLeft) = [];
end

% Fill cropped images to make masks for differencing
for i = 1:2
    
    if i == 1
        imhalf = padarray(Icrop1, [1 1], 1);
    else
        imhalf = padarray(Icrop2, [1 1], 1);
    end

    if strcmp(es,'SN') || strcmp(es,'NS') || strcmp(es,'NW') || strcmp(es,'WN') ||  strcmp(es,'SW') || strcmp(es,'WS')
        imhalf(:,end) = [];
        imhalf = imfill(imhalf, 'holes');
        imhalf(:,1) = [];
        imhalf(end,:) = [];
        imhalf(1,:) = [];
    elseif strcmp(es,'EW') || strcmp(es,'WE')
        imhalf(end,:) = [];
        imhalf = imfill(imhalf, 'holes');
        imhalf(:,1) = [];
        imhalf(:,end) = [];
        imhalf(1,:) = [];
    elseif strcmp(es,'NE') || strcmp(es,'EN') || strcmp(es,'SE') || strcmp(es,'ES')
        imhalf(:,1) = [];
        imhalf = imfill(imhalf, 'holes');
        imhalf(:,end) = [];
        imhalf(1,:) = [];
        imhalf(end,:) = [];       
    end
    imcrop{i} = imhalf;
end

% Find image of migrated centerline areas
mig_areas = imsubtract(imcrop{1},imcrop{2}) | imsubtract(imcrop{2},imcrop{1});

% Ensure that migrated area does not include the cl at t1 and does include
% the cl at t2
mig_areas(Icrop2) = true; % xy2
mig_areas(Icrop1) = false; % xy1

% Put migrated areas back in place relative to original image size (if
% provided)
Imig = false(sizeI);
if isempty(yTop) == 1
    yTop = 0;
end
if isempty(xLeft) == 1
    xLeft = 0;
end
Imig(yTop+1:size(mig_areas,1)+yTop,xLeft+1:size(mig_areas,2)+xLeft) = mig_areas;

%% Find cutoffs
% Find cutoff locations along xy1 by thresholding the along stream distance
% between intersections of xy1 and xy2. If the distances are large between
% any two intersection points, this indicates channel shortening (cutoff).

% Set the threshold length of channel shortening to identify cutoffs
cutthreshlen = W*2;

% Find indices of intersections between xy1, xy2
[~,~,iout,jout] = intersections(xy1(:,1),xy1(:,2),xy2(:,1),xy2(:,2),1);
intarray = [iout, jout];
% Remove NaNs (byproduct of intersections.m function)
rem = find(isnan(intarray(:,1)));
intarray(rem,:) = [];
% Sort intersections according to streamwise distance
intarray = sortrows(intarray,1);
% Sorted intersection indices
iout = intarray(:,1);
jout = intarray(:,2);
% Compute streamwise distances between each node along both xy1 and xy2
ds1 = [sqrt((diff(xy1(:,1))).^2+(diff(xy1(:,2))).^2); 0];
ds2 = [sqrt((diff(xy2(:,1))).^2+(diff(xy2(:,2))).^2); 0];
% Find length of stream between each intersection
for i = 1:numel(iout)-1
    seglen1(i) = sum(ds1(floor(iout(i)):floor(iout(i+1))));        
    seglen2(i) = sum(ds2(floor(jout(i)):floor(jout(i+1))));
end

% Find cutoffs by thresholding the lengths between intersections
int_idcs = find(seglen1-seglen2>cutthreshlen);

% Initialize cutoff image, area storage, cutoff indices
Icutsall = false(sizeI);
cutarea = [];
cutidcs = [];
chutelen = [];
cutlen = [];

% If no cutoffs are found, halt
if isempty(int_idcs)
    return
end

% Loop through all cutoffs to find them in Imig and compute their
% properties, location
for jj = 1:numel(int_idcs)
    % Cutoff indices
    usidx = int_idcs(jj);
    dsidx = int_idcs(jj)+1;
    % Check next downstream and upstream intersections to make sure
    % entirety of cutoff was captured. R6, i = 13 is an example case
    % demonstrating the necessity of this step.   
    for us = 1:2
        if us == 1 % upstream check
            xy1d = iout(int_idcs(jj));
            xy2d = jout(int_idcs(jj));
            xy1u = iout(int_idcs(jj)-1);
            xy2u = jout(int_idcs(jj)-1);
        elseif us == 2 % downstream check
            xy1u = iout(int_idcs(jj)+1);
            xy2u = jout(int_idcs(jj)+1);
            xy1d = iout(int_idcs(jj)+2);
            xy2d = jout(int_idcs(jj)+2);
        end            
        pgonx = [xy1(xy1u:xy1d,1); flipud(xy2(xy2u:xy2d,1))];
        pgony = [xy1(xy1u:xy1d,2); flipud(xy2(xy2u:xy2d,2))];
        A(us) = polyarea(pgonx,pgony);
    end
    if A(1) > W^2*5
        usidx = int_idcs(jj)-1;
    end
    if A(2) > W^2*5
        dsidx = int_idcs(jj)+2;
    end
    
    % Polygon coordinates of cutoff
    xy1cut = xy1(round(iout(usidx)):round(iout(dsidx)),:);        
    xy2cut = xy2(round(jout(usidx)):round(jout(dsidx)),:);

    % Extract the coordinates from xy1, xy2 portions that encompass cutoffs 
    xs = [xy1cut(:,1); xy2cut(:,1)];
    ys = [xy1cut(:,2); xy2cut(:,2)];
    
    % Create a mask with these coordinates
    Icutmask = poly2mask(xs, ys, sizeI(1), sizeI(2));
    
    % Dilate the mask slightly so the xy2 centerline is included
    Icutmask = bwmorph(Icutmask,'dilate',1);
    
    % Use the mask to find cutoff area from Imig
    Icut = Imig & Icutmask;
    
    % Remove "tails" from cutoff areas
    % Create mask of cutoff areas for tail removal
    Iremtails = Icut;
    
    % Erode cutoff areas mask to remove any tails that are actual migration
    Iremtails = bwmorph(Iremtails,'erode',5);
    
    % Over-dilate cutoff areas mask to ensure all cutoff area is masked
    Iremtails = bwmorph(Iremtails,'dilate',10);
    
    % Apply the mask
    Icutnotails = Icut & Iremtails;
    
    % Keep only the largest blob (in case small patches remain)
    rp = regionprops(Icutnotails,'Area','PixelIdxList');
    [~,maxidx] = max([rp(:).Area]);
    Icutnotails = false(size(Icutnotails));
    Icutnotails(rp(maxidx).PixelIdxList) = true;

    % Store the cutoff
    Icutsall(Icutnotails) = true;
    
    % Remove cutoff from migrated area image
    Imig(Icutnotails) = false;
    
    % Compute cutoff area
    cutarea(jj,1) = sum(sum(Icutnotails));
    
    % Find the indices nearest the border of the tail-removed cutoff
    % Make image with xy1 line (for computing length)
    Ilen = false(sizeI);
    idcs = sub2ind(size(Ilen),round(xy1(:,2)),round(xy1(:,1)));
    Ilen(idcs) = true;
    
    % Remove cutoffs from xy1 image
    Icutdilate = bwmorph(Icutnotails,'dilate',1); % must dilate to re-capture xy1 since it was not included in migrated area
    Ilencut = Ilen & Icutdilate;
    Ilencut = bwareaopen(Ilencut, 10); % remove small segments (in case not all of centerline was removed after masking with Icutdilate)
    
    % Compute length of non-cutoff portions of river
    lenprops = regionprops(Ilencut,'Perimeter');   % Not exact length
    nocutlen = sum([lenprops(:).Perimeter])/2;
    
    % Find centerline indices of cutoffs (indices are w.r.t. xy1)
    Ilencut = bwmorph(Ilencut,'skel',Inf); % ensure skeleton
    
    % Find endpoints of xy1 cutoff image
    [yac, xac] = find(bwmorph(Ilencut, 'endpoints'));
    maxdis = [];
    for zz = 1:numel(yac)
        D = bwdistgeodesic(Ilencut,xac(zz),yac(zz));
        maxdis(zz) = max(max(D));
    end
    keepidcs = find(maxdis==max(maxdis));
    endpts = [xac(keepidcs) yac(keepidcs)];
    
    % Find indices of cutoff endpoints
    [~,~,ib] = intersect(endpts, round(xy1), 'rows');
    cutidcs(jj,:) = [sort(ib)]';
    
    % Cutoff length
    cutlen(jj) = sum(ds1(floor(cutidcs(jj,1)):floor(cutidcs(jj,2))));
    
    % Compute length of connecting chute
    [~,i1] = min(sqrt((xy1(floor(cutidcs(jj,1)),1)-xy2(:,1)).^2+(xy1(floor(cutidcs(jj,1)),2)-xy2(:,2)).^2));
    [~,i2] = min(sqrt((xy1(floor(cutidcs(jj,2)),1)-xy2(:,1)).^2+(xy1(floor(cutidcs(jj,2)),2)-xy2(:,2)).^2));
    chutelen(jj) = sum(ds2(i1:i2));    
end