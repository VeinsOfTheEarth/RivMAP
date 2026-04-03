function [cl, Icl] = centerline_from_mask(I, es, W, plotornot)

% Computes a centerline from an input binary channel mask. If the mask is a
% multi-threaded channel, the "holes" between channel threads are filled
% and the returned centerline will be relative to the channel extents
% rather than individual threads.

% Requires the Image Processing toolbox

% INPUTS:      I - binary single-thread image of river (river mask)
%             es - two-element string specifying the sides of the image
%                  that the channel intersects (e.g. 'SW', 'NS', etc.).
%                  The first character should be the upstream
%                  intersection.
%              W - nominal channel width, used to set the size of the
%                  padded mirrored boundaries
%      plotornot - (optional) A value of '1' will plot results.
%
% OUTPUTS:     cl - Nx2 vector of channel coordinates arranged in
%                   upstream->downstream order
%             Icl - image of centerline pixels

% keyboard

if nargin == 3
    plotornot = 0;
end

if sum(sum(I)) < 100
    disp('No mask detected in input image')
    cl = [];
    return
end

% Fill any holes
Io = I;
I = imfill(I,'holes');
sizeIo = size(I);

% Crop the mask so channel intersects image boundaries
[I,yTop,yBottom,xLeft,xRight] = crop_to_mask(I,es);

[nr, nc] = size(I);

% Extend the ends of the mask using mirroring so that the centerline ends
% can be resolved
% Create image of original mask combined with mirrored images
Imirror = rot90(I,2);
Icombined = false(size(I)*3);
Icombined(nr:nr*2-1,nc:nc*2-1) = I;
% Fill in the combined image with mirrored pads
for i = 1:2
    if strcmp(es(i),'N')
        idx_I = find(Icombined(nr,:)==1,1,'first');
        idx_Im = find(Imirror(end,:)==1,1,'first');
        Icombined(1:nr,idx_I-idx_Im:idx_I-idx_Im+nc-1) = Imirror; 
    elseif strcmp(es(i),'E')
        idx_I = find(Icombined(:,2*nc-1)==1,1,'first');
        idx_Im = find(Imirror(:,1)==1,1,'first');
        Icombined(idx_I-idx_Im:idx_I-idx_Im+nr-1,2*nc:3*nc-1) = Imirror; 
    elseif strcmp(es(i),'S')
        idx_I = find(Icombined(2*nr-1,:)==1,1,'first');
        idx_Im = find(Imirror(1,:)==1,1,'first');
        Icombined(2*nr:3*nr-1,idx_I-idx_Im:idx_I-idx_Im+nc-1) = Imirror; 
    elseif strcmp(es(i),'W')
        idx_I = find(Icombined(:,nc)==1,1,'first');
        idx_Im = find(Imirror(:,end)==1,1,'first');
        Icombined(idx_I-idx_Im:idx_I-idx_Im+nr-1,1:nc) = Imirror; 
    end
end
% Crop the combined, padded image
padwidth = round(max(sizeIo)/5);
I = Icombined(nr-padwidth:nr*2-1+padwidth,nc-padwidth:nc*2-1+padwidth);

% Fill any holes again (in case some appeared due to mirroring)
I = imfill(I,'holes');

% Remove any spurious small patches by keeping only the largest blob
sp = regionprops(I,'Area','PixelIdxList');
[~,maxidx] = max([sp(:).Area]);
I = false(size(I));
I(sp(maxidx).PixelIdxList) = true;

% Uncomment for smoothing the binary mask
% se = strel('disk',15);   
% I = imclose(I,se);       

% Perform skeletonization
Iskel = bwmorph(I,'skel',Inf);
Iskel = bwmorph(Iskel,'thin');
Iskel = bwareaopen(Iskel, 10); % In some cases, single isolated pixels remain in the skeleton. This removes such instances.

% Find endpoints of centerline
E = bwmorph(Iskel, 'endpoints');
[yE,xE] = find(E);
for i = 1:2
    if strcmp(es(i),'N')
        [~,epidx(i)] = min(yE);
    elseif strcmp(es(i),'E')
        [~,epidx(i)] = max(xE);
    elseif strcmp(es(i),'S')
        [~,epidx(i)] = max(yE);
    elseif strcmp(es(i),'W')
        [~,epidx(i)] = min(xE);
    end
end
endpts(1,:) = [yE(epidx(1)),xE(epidx(1))];
endpts(2,:) = [yE(epidx(2)),xE(epidx(2))];

% Find centerline skeleton
D1 = bwdistgeodesic(Iskel, endpts(1,2), endpts(1,1));
D2 = bwdistgeodesic(Iskel, endpts(2,2), endpts(2,1));
Dadd = D1 + D2;
Dadd = round(Dadd * 8) / 8;
Dadd(isnan(Dadd)) = inf;
Iskel = imregionalmin(Dadd);
Iskel = bwmorph(Iskel,'thin'); 

% Remove mirrored padding from skeleton image
Iskel = Iskel(padwidth:nr+padwidth-1,padwidth:nc+padwidth-1);
Iskel = bwareaopen(Iskel,W*5);

% Put skeleton back into uncropped original reference frame
Icl = false(sizeIo);
Icl(yTop:yTop+nr-1,xLeft:xLeft+nc-1) = Iskel;

% Find endpoints of binary centerline
[yend, xend] = find(bwmorph(Icl,'endpoints'));    

% Find upstream->downstream ordered centerline coordinates
cl = skeleton_coords(Icl,xend(2),yend(2),es);

% Plotting
if plotornot
    figure
    imshow(Io); hold on
    plot(cl(:,1),cl(:,2),'r','linewidth',1.5)
end

%% Code graveyard - three other methods for spur removal

% %% Old method for removing spurs
% % Remove remaining branches
% % Restrict to end regions first
% B = bwmorph(Iskel, 'branchpoints');
% E = bwmorph(Iskel, 'endpoints');
% [yB,xB] = find(B);
% [yE,xE] = find(E);
% B_loc = find(B);
% 
% % Identify the true centerline endpoints by thresholding the
% % distance between branchpoints near the image boundaries to remove spurs
% % connected to the centerline endpoint
% keyboard
% for i = 1:2
%     if strcmp(es(i),'N')
%         Bends = find(yB<W);
%     elseif strcmp(es(i),'E')
%         Bends = find(xB>size(I,2)-W);
%     elseif strcmp(es(i),'S')
%         Bends = find(yB>size(I,1)-W);
%     elseif strcmp(es(i),'W')
%         Bends = find(xB<W);
%     end
%     
%     % Remove the branchpoint pixels that are near to each other, along with
%     % their neighbors to destroy connectivity
%     for jj = 1:numel(Bends)
%         Iskel(yB(Bends(jj))-1:yB(Bends(jj))+1,xB(Bends(jj))-1:xB(Bends(jj))+1) = false;
%     end    
% end
% % Keep only largest blob now that spurious centerline ends have been
% % detached
% cc = bwconncomp(Iskel);
% for i = 1:cc.NumObjects
%     ccsize(i) = numel(cc.PixelIdxList{i});
% end
% [~,maxidx] = max(ccsize);
% Iskel = false(size(Iskel));
% Iskel(cc.PixelIdxList{maxidx}) = true;
% 
% % % Remove most of the branches with spur
% % Iskel = bwmorph(Iskel,'spur','Inf');
% 
% % Now the extreme endpoints of the skeleton should be the true centerline
% % endpoints, so identify them (spurs still attached)
% E = bwmorph(Iskel, 'endpoints');
% [yE,xE] = find(E);
% for i = 1:2
%     if strcmp(es(i),'N')
%         [~,epidx(i)] = min(yE);
%     elseif strcmp(es(i),'E')
%         [~,epidx(i)] = max(xE);
%     elseif strcmp(es(i),'S')
%         [~,epidx(i)] = max(yE);
%     elseif strcmp(es(i),'W')
%         [~,epidx(i)] = min(xE);
%     end
% end
% endpts(1,:) = [yE(epidx(1)),xE(epidx(1))];
% endpts(2,:) = [yE(epidx(2)),xE(epidx(2))];
% 
% % Find the skeleton between the endpoints (bypasses spurs)
% D1 = bwdistgeodesic(Iskel, endpts(1,2), endpts(1,1));
% D2 = bwdistgeodesic(Iskel, endpts(2,2), endpts(2,1));
% Dadd = D1 + D2;
% Dadd = round(Dadd * 8) / 8;
% Dadd(isnan(Dadd)) = inf;
% Iskel = imregionalmin(Dadd);
% Iskel = bwmorph(Iskel,'thin'); 


        
% %% Older method for removing spurs
% n_in_shrink = 20;
% % Perform initial shrink
% Iskel = bwmorph(Iskel,'shrink',n_in_shrink);
% % Continue shrinking until there are no more branchpoints
% count = n_in_shrink;
% while count > -1
%     B = bwmorph(Iskel, 'branchpoints');
%     nB = find(B);
%     if isempty(nB)
%         break
%     end
%     Iskel = bwmorph(Iskel,'shrink',1);
%     count = count + 1;
% end
% 
% % Add back the centerline on the ends that was removed by the shrinking
% % process. This is done by adding back the two pieces with the extreme-most
% % coordinates w.r.t. the exit direction.
% 
% % Shrink once to remove cases where ends of centerline skeleton produce
% % extra branchpoint
% skel_diff = Iskelsave & ~Iskel;
% 
% % Find x,y locations of uncleaned skeleton
% skel_pixels = regionprops(Iskelsave,'PixelList');
% % Find x,y locations of the removed pieces
% group_pixels = regionprops(skel_diff,'PixelList','PixelIdxList');
% % Find which two groups to keep
% for s = 1:2 % For each end of the centerline
%     
%     if es(s) == 'N'
%         
%         minY = min(skel_pixels.PixelList(:,2));
%         for ss = 1:numel(group_pixels)
%             if sum(group_pixels(ss).PixelList(:,2) == minY)>0;
%                 gkeep(s) = ss;
%                 break
%             end
%         end
%             
%     elseif es(s) == 'S'
%         
%         maxY = max(skel_pixels.PixelList(:,2));
%         for ss = 1:numel(group_pixels)
%             if sum(group_pixels(ss).PixelList(:,2) == maxY)>0;
%                 gkeep(s) = ss;
%                 break
%             end
%         end
% 
%     elseif es(s) == 'W'
%         
%         minX = min(skel_pixels.PixelList(:,1));
%         for ss = 1:numel(group_pixels)
%             if sum(group_pixels(ss).PixelList(:,1) == minX)>0;
%                 gkeep(s) = ss;
%                 break
%             end
%         end
% 
%     elseif es(s) == 'E'
%         
%         maxX = max(skel_pixels.PixelList(:,1));
%         for ss = 1:numel(group_pixels)
%             if sum(group_pixels(ss).PixelList(:,1) == maxX)>0;
%                 gkeep(s) = ss;
%                 break
%             end
%         end
%     end
% 
% end % for 
% 
% % % Remove spurs from centerline ends
% % I1 = false(size(I));
% % I2 = I1;
% 
% % Add groups back to skeletonized image
% Iskel(group_pixels(gkeep(1)).PixelIdxList) = true;
% Iskel(group_pixels(gkeep(2)).PixelIdxList) = true;
% 
% % Check once more in case we added back any spurs
% count2 = 0;
% while count2 > -1
%     B = bwmorph(Iskel, 'branchpoints');
%     nB = find(B);
%     if isempty(nB)
%         break
%     end
%     Iskel = bwmorph(Iskel,'shrink',1);
%     count2 = count2 + 1;
% end



% %% Oldest method for removing spurs
% % % Remove spurious links; see the following link for logic.
% % % http://www.mathworks.com/matlabcentral/answers/88284-remove-the-spurious-edge-of-skeleton
% count = 0;
% while count > -1
%     
%     B = bwmorph(Iskel, 'branchpoints');
%     E = bwmorph(Iskel, 'endpoints');
% 
%     [yE,xE] = find(E);
%     B_loc = find(B);
%     
%     % Once the spurs at the end of the centerline have been removed, we
%     % don't want to consider the actual centerline endpoints (This is not
%     % robust but seems to work)
%     if count > 0
%         xE(1) = []; yE(1) = [];
%         xE(end) = []; yE(end) = [];
%     end
%     
%     if isempty(B_loc) % stop when only the endpoints remain
%        break;
%     end
%     
%     [yB, xB] = find(B);
%     imshow(Iskel); hold on
%     plot(xE,yE,'om')
%     plot(xB,yB,'oc')
% 
%     Dmask = false(size(Iskel));
%     for k = 1:numel(xE)
%         D = bwdistgeodesic(Iskel,xE(k),yE(k));
%         distanceToBranchPt = min(D(B_loc));
%         Dmask(D < distanceToBranchPt) = true;
%     end
%     Iskel_new = Iskel & ~Dmask;
%     Iskel_new = bwmorph(Iskel_new,'skel',Inf);
%     Iskel = Iskel_new;
%     
%     count = count + 1;
% 
% end % while
% Iskel = bwmorph(Iskel,'bridge');    % in case there are any single-pixel gaps in the centerline


