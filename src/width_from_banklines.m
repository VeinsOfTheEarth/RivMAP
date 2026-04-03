function width = width_from_banklines(cl, lb, rb, W)

% Computes channel width at all points along a centerline by intersecting
% perpendicular vectors with the left and right banks.


% INPUTS:      cl - Nx2 array of x,y centerline coordinates
%              lb - Lx2 array of x,y left bank coordinates
%              rb - Rx2 array of x,y right bank coordinates
%              W  - nominal channel width (in units of cl,lb,rb--pixels)
%                   used for parameterizing distance to search for 
%                   intersections with bank
%
% OUTPUTS:     width - average channel width at each segment

% keyboard

% Smooth the centerline coordinates
cl = savfilt(cl,W);

dx = W*3;   % how far away from centerline to search for intersection with bank
% Initialize storage
width = nan(length(cl),1);

% First centerline point width will be NaN.
ncrop = 100; % number of indices away from cl pt to crop banks for intersection search
L = length(lb);
R = length(rb);

% parfor a = 2:length(cl)-1;
for a = 2:length(cl)-1;
        
    tpx = cl(a,1);
    tpy = cl(a,2);
    % Slope of line from current centerline point to point on circle
    m = (tpy-cl(a-1,2))/(tpx-cl(a-1,1));
    % Slope of perpendicular line
    minv = -1/m;
    % Construct perpendicular lines
    upper_pt = [tpx + dx, tpy + dx*minv];
    lower_pt = [tpx - dx, tpy - dx*minv];
    perpx = [upper_pt(1) lower_pt(1)];
    perpy = [upper_pt(2) lower_pt(2)];
    
%     clf
%     plot(lb(:,1),lb(:,2),'r'); hold on; axis equal
%     plot(rb(:,1),rb(:,2),'r');
%     plot(cl(:,1),cl(:,2),'b');
%     plot(tpx,tpy,'.k')
%     plot(cl(a-1,1),cl(a-1,2),'k.')
%     plot(perpx,perpy,'b')

    % Crop banklines for faster intersections search
    % Find index of bank points closest to cl pt
    [~,lbclosestidx] = min(sqrt((lb(:,1)-tpx).^2+(lb(:,2)-tpy).^2));
    [~,rbclosestidx] = min(sqrt((rb(:,1)-tpx).^2+(rb(:,2)-tpy).^2));

    if lbclosestidx > ncrop && rbclosestidx > ncrop && ...
        ncrop < L-lbclosestidx-1 && ncrop < R-rbclosestidx-1
        lbcrop = lb(lbclosestidx-ncrop+1:lbclosestidx+ncrop-1,:);
        rbcrop = rb(rbclosestidx-ncrop+1:rbclosestidx+ncrop-1,:);
    else
        lbcrop = lb;
        rbcrop = rb;
    end
        
    [xl,yl,~,~] = intersections(perpx,perpy,lbcrop(:,1),lbcrop(:,2),1);
    [xr,yr,~,~] = intersections(perpx,perpy,rbcrop(:,1),rbcrop(:,2),1);
    nl = numel(xl); % number of left-bank intersections
    nr = numel(xr); % number of right-bank intersections
    if  nl == 0 || nr == 0   % Skip if lines don't intersect both banks - either increase dx or accept NaN value
    else
        % In case multiple intersections are returned, use the closest
        center_to_lb = sqrt((xl-cl(a,1)).^2+(yl-cl(a,2)).^2);
        center_to_lb = min(center_to_lb);
        center_to_rb = sqrt((xr-cl(a,1)).^2+(yr-cl(a,2)).^2);
        center_to_rb = min(center_to_rb);
        width(a) = center_to_lb + center_to_rb;
    end
end % a loop

