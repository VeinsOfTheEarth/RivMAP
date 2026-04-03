function [Ie,Ia,Inc,Icuts] = migration_mask(I1,I2,W,Icut)

% Computes erosion, accretion, no-change, and cutoff images by 
% mask-differencing two binary channel masks from different times. 
% Cutoffs are identified one of two ways. If the image of cutoffs
% found from migration_cl is input, blobs of accretion that intersect any
% cutoffs are removed. Otherwise, simple area-thresholding is performed.

% Requires the Image Processing toolbox

% INPUTS:      I1 - binary mask of channel at time 1
%              I2 - binary mask of channel at time 2
%               W - nominal channel width, used for thresholding cutoff
%                   size
% (optional) Icut - image of cutoff areas identified from
%                   centerline analysis (output of migration_cl)
%
% OUTPUTS:     Ie - binary image of eroded areas
%              Ia - binary image of accreted areas
%             Inc - binary image of no-change areas
%           Icuts - binary image of cutoff areas

% Difference the masks
Idiff = I1-I2;

% Find erosion and accretion and no change images
Ie = false(size(Idiff));
Ia = Ie;
Inc = Ie;

Ie(Idiff==-1) = true;   % Erosion image
Ia(Idiff==1) = true;    % Accretion image
Inc(Idiff==0) = true;   % No change image

% Identify cutoffs via area-thresholding if no cutoff image is provided
if nargin < 4
    % Identify cutoffs by thresholding blob sizes of the accretion image
    cc = bwconncomp(Ia);
    rp = regionprops(cc,'Area');
    areas = [rp.Area];

    % Threshold blob areas
    a_thresh = 2*W^2;
    cutidcs = find(areas>a_thresh);

    % Make image of cutoff areas and remove cutoff areas from accretion image
    Icuts = false(size(I1));
    for j = 1:numel(cutidcs)
        Icuts(cc.PixelIdxList{cutidcs(j)}) = true;
        Ia(cc.PixelIdxList{cutidcs(j)}) = false;
    end
else % Identify cutoffs by intersecting provided cutoff image with accretion blobs
    % Find blobs of potential cutoffs from accretion image
    cc = bwconncomp(Ia);
    rp = regionprops(cc,'Area','PixelIdxList');
    
    % Find accreted area blobs that overlap with Icut
    cutpixelidx = find(Icut);
    cutrem = zeros(numel(rp),1);
    for i = 1:numel(rp)
        cutrem(i) = sum(ismember(rp(i).PixelIdxList, cutpixelidx));
    end
    remidcs = find(cutrem>0);
    
    % Make image of cutoff areas
    Icuts = false(size(I1));
    for i = 1:numel(remidcs)
        Icuts(rp(remidcs(i)).PixelIdxList) = true;
    end
end  
