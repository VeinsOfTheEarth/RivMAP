function [Icrop,yTop,yBottom,xLeft,xRight] = crop_to_mask(Icrop,es)

% Crops an input image to its extents (i.e. removes columns/rows containing
% all zeros), then checks that the mask has no holes at the intersections 
% with both image edges. 

% Requires the Image Processing toolbox

% INPUTS:      Icrop - image to be cropped
%                 es - "exit sides", a two-letter string array from NESW 
%                      options (e.g. 'NS') where the first letter represents 
%                      the side of the image that the downstream end of the 
%                      river enters the image and the second letter represents
%                      the side of the image where the river exits.
%
% OUTPUTS:     Icrop - cropped image
%               yTop - row that cropping ends at top of image
%            yBottom - row that cropping ends at bottom of image
%              xLeft - column that cropping ends at left of image
%             xRight - column that cropping ends at right of image


% Original image size
sizeIo = size(Icrop);

% Find x,y coordinates of mask boundaries
boundaries = cell2mat(bwboundaries(Icrop));
x = boundaries(:,2);  
y = boundaries(:,1);  

% Perform cropping; order of cropping is important (SE, then NW)
if ismember('S',es) == 1
    yBottom = ceil(max(y));
    Icrop(yBottom+1:end,:)=[];
else
    yBottom = sizeIo(1);
end
if ismember('E',es) == 1
    xRight = ceil(max(x));
    Icrop(:,xRight+1:end) = [];
else
    xRight = sizeIo(2);
end
if ismember('N',es) == 1
    yTop = ceil(min(y));
    Icrop(1:yTop-1,:)=[];
else
    yTop = 1;
end
if ismember('W',es) == 1
    xLeft = ceil(min(x));
    Icrop(:,1:xLeft-1) = [];
else
    xLeft = 1;
end

% Fill in any holes in the edges of the binary mask
% Bottom
Icrop(end+1,:) = true;
Icrop = imfill(Icrop,'holes');
% Top
Icrop(2:end,:) = Icrop(1:end-1,:);
Icrop(1,:) = true;
Icrop = imfill(Icrop,'holes');
Icrop(1,:) = [];
% Right
Icrop(:,end+1) = true;
Icrop = imfill(Icrop,'holes');
% Left
Icrop(:,2:end) = Icrop(:,1:end-1);
Icrop(:,1) = true;
Icrop = imfill(Icrop,'holes');
Icrop(:,1) = [];
