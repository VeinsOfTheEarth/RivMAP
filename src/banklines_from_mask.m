function [banksout] = banklines_from_mask(I,es,plotornot)

% Returns banklines from a binary channel mask.

% Requires the Image Processing toolbox
% 
% INPUTS:      I  - cleaned, binary image
%              es - "exit sides", a two-letter string array from NESW 
%                   options (e.g. 'NS') where the first letter represents 
%                   the side of the image that the downstream end of the 
%                   river enters the image and the second letter represents
%                   the side of the image where the river exits.
%       plotornot -  (optional) plot image with banks overlaid 1=yes
%              
% OUTPUTS:  banksout - 2x1 cell; first element is left bank, second is
%                      right bank. Each cell contains a 2-column array of x
%                      and y coordinates. x,y coordinates are returned with 
%                      respect to Matlab's image plotting coordinates; 
%                      namely, the origin is at the top-left of the image,
%                      not the bottom left. (Must multiply y-axis by -1 for 
%                      "regular" plotting unless plotting on top of an
%                      image.) Coordinates are returned in an 
%                      upstream-to-downstream orientation.

if nargin == 2
    plotornot = 0;
else
    Io = I; % Save original image for plotting
end

% Make sure input mask exists
if sum(sum(I)) < 100
    disp('No mask detected in input image')
    banksout{1} = [];
    banksout{2} = [];
    return
end

% Fill any holes
I = imfill(I,'holes');

% Remove any spurious small patches by keeping only largest blob
sp = regionprops(I,'Area','PixelIdxList');
[~,maxidx] = max([sp(:).Area]);
I = false(size(I));
I(sp(maxidx).PixelIdxList) = true;

% Crop mask to edges of image
[Icropped,yTop,yBottom,xLeft,xRight] = crop_to_mask(I,es);

% Find perimeter of mask
Iperim = bwperim(Icropped);
Ibanks = Iperim;

% Remove unwanted boundary pixels by removing endpoints
E = find(bwmorph(Ibanks,'endpoints'));
count = 0;
while numel(E)>0
    Ibanks(E) = false;
    E = find(bwmorph(Ibanks,'endpoints'));
    count = count + 1;
    if count > 25 % Try removing extra pixels with spur, skel commands
%         Ibanks = imfill(Ibanks,'holes');
        Ibanks = bwmorph(Ibanks,'skel','Inf');
        Ibanks = bwmorph(Ibanks,'spur',1);
        break
    end
end

% Erase boundaries that are not part of banklines
% Lots of code required for handling a few boundary pixels
for i = 1:2
    if strcmp(es(i),'E') == 1
        % Remove border pixels
        Ibanks(:,end) = false;
        % Remove unattached groups
        Ibanks = bwareaopen(Ibanks,20);
        % Add proper border pixels back to perimeter
        [yp, xp] = find(Ibanks);
        x = size(Ibanks,2)-1;
        edgeminusonevals = yp(xp==x);
        diffs = diff(edgeminusonevals);
        [~,maxidx] = max(abs(diffs));
        miny = edgeminusonevals(maxidx);
        maxy =(edgeminusonevals(maxidx+1));
        for mm = 1:2
            if mm == 1
                y = maxy;
                if Iperim(y+1,x+1) == 1
                    Ibanks(y+1,x+1) = 1;
                elseif Iperim(y,x+1) == 1
                    Ibanks(y,x+1) = 1;
                elseif Iperim(y-1,x+1) == 1
                    Ibanks(y-1,x+1) = 1;
                end
            else
                y = miny;
                if Iperim(y-1,x+1) == 1
                    Ibanks(y-1,x+1) = 1;
                elseif Iperim(y,x+1) == 1
                    Ibanks(y,x+1) = 1;
                elseif Iperim(y+1,x+1) == 1
                    Ibanks(y+1,x+1) = 1;
                end
            end
        end
    elseif strcmp(es(i),'W') == 1
        % Remove border pixels
        Ibanks(:,1) = false;
        % Remove unattached groups
        Ibanks = bwareaopen(Ibanks,20);
        % Add proper border pixels back to perimeter
        [yp, xp] = find(Ibanks);
        x = 2;
        edgeminusonevals = yp(xp==x);
        diffs = diff(edgeminusonevals);
        [~,maxidx] = max(abs(diffs));
        miny = edgeminusonevals(maxidx);
        maxy =(edgeminusonevals(maxidx+1));
        for mm = 1:2
            if mm == 1
                y = maxy;
                if Iperim(y+1,x-1) == 1
                    Ibanks(y+1,x-1) = 1;
                elseif Iperim(y,x-1) == 1
                    Ibanks(y,x-1) = 1;
                elseif Iperim(y-1,x-1) == 1
                    Ibanks(y-1,x-1) = 1;
                end
            else
                y = miny;
                if Iperim(y-1,x-1) == 1
                    Ibanks(y-1,x-1) = 1;
                elseif Iperim(y,x-1) == 1
                    Ibanks(y,x-1) = 1;
                elseif Iperim(y+1,x-1) == 1
                    Ibanks(y+1,x-1) = 1;
                end
            end
        end
    elseif strcmp(es(i),'N') == 1
        % Remove border pixels
        Ibanks(1,:) = false;
        % Remove unattached groups
        Ibanks = bwareaopen(Ibanks,20);
        % Add proper border pixels back to perimeter
        [yp, xp] = find(Ibanks);
        y = 2;
        edgeminusonevals = xp(yp==y);
        diffs = diff(edgeminusonevals);
        [~,maxidx] = max(abs(diffs));
        minx = edgeminusonevals(maxidx);
        maxx = edgeminusonevals(maxidx+1);
        for mm = 1:2
            if mm == 1
                x = maxx;
                if Iperim(y-1,x+1) == 1
                    Ibanks(y-1,x+1) = 1;
                elseif Iperim(y-1,x) == 1
                    Ibanks(y-1,x) = 1;
                elseif Iperim(y-1,x-1) == 1
                    Ibanks(y-1,x-1) = 1;
                end
            else
                x = minx;
                if Iperim(y-1,x-1) == 1
                    Ibanks(y-1,x-1) = 1;
                elseif Iperim(y-1,x) == 1
                    Ibanks(y-1,x) = 1;
                elseif Iperim(y-1,x+1) == 1
                    Ibanks(y-1,x+1) = 1;
                end
            end
        end
    elseif strcmp(es(i),'S') == 1
        % Remove border pixels
        Ibanks(end,:) = false;
        % Remove unattached groups
        Ibanks = bwareaopen(Ibanks,20);
        % Add proper border pixels back to perimeter
        [yp, xp] = find(Ibanks);
        y = size(Ibanks,1)-1;        
        edgeminusonevals = xp(yp==y);
        diffs = diff(edgeminusonevals);
        [~,maxidx] = max(abs(diffs));
        minx = edgeminusonevals(maxidx);
        maxx = edgeminusonevals(maxidx+1);        
       for mm = 1:2
            if mm == 1
                x = maxx;
                if Iperim(y+1,x+1) == 1
                    Ibanks(y+1,x+1) = 1;
                elseif Iperim(y+1,x) == 1
                    Ibanks(y+1,x) = 1;
                elseif Iperim(y+1,x-1) == 1
                    Ibanks(y+1,x-1) = 1;
                end
            else
                x = minx;
                if Iperim(y+1,x-1) == 1
                    Ibanks(y+1,x-1) = 1;
                elseif Iperim(y+1,x) == 1
                    Ibanks(y+1,x) = 1;
                elseif Iperim(y+1,x+1) == 1
                    Ibanks(y+1,x+1) = 1;
                end
            end
        end
    end
end

% Extract each bank 
ccbanks = bwconncomp(Ibanks);
% parfor lr = 1:2
for lr = 1:2
    % Make banks into image
    Itemp = false(size(Ibanks));
    Itemp(ccbanks.PixelIdxList{lr}) = true;
      
    % Find "true" endpoints of centerline
    [ptsy,ptsx] = find(Itemp);
    E = [find(ptsy==1); find(ptsy==size(Itemp,1)); find(ptsx==1); find(ptsx==size(Itemp,2))];
    Eind = sub2ind(size(Itemp),ptsy(E),ptsx(E));
           
    if numel(Eind) ~= 2
        disp(['Not two endpoints, lr=',num2str(lr)])
    end

    % Find shortest path between endpoints (cuts off spurious loops along bank edge)
    D1 = bwdistgeodesic(Itemp,Eind(1));
    D2 = bwdistgeodesic(Itemp,Eind(2));
    Dadd = D1 + D2;
    Dadd = round(Dadd * 8) / 8;
    Dadd(isnan(Dadd)) = inf;
    Itemp = imregionalmin(Dadd);
    Itemp = bwmorph(Itemp,'thin'); 
            
    % Trace skeleton to find bankline coordinates
    xy = skeleton_coords(Itemp,ptsx(E(1)),ptsy(E(1)),es);
    
    % Adjust coordinates to original reference frame
    xy(:,1) = xy(:,1) + xLeft - 1;
    xy(:,2) = xy(:,2) + yTop - 1;
  
    banks{lr} = xy;
end

% Find which bank is left and which is right
banksout{2} = [];
if es(1) == 'S'
    if banks{1}(1,1) < banks{2}(1,1)
        banksout{1} = banks{1};
        banksout{2} = banks{2};
    else
        banksout{1} = banks{2};
        banksout{2} = banks{1};
    end
elseif es(1) == 'N'
    if banks{1}(1,1) > banks{2}(1,1)
        banksout{1} = banks{1};
        banksout{2} = banks{2};
    else
        banksout{1} = banks{2};
        banksout{2} = banks{1};
    end
elseif es(1) == 'E'
    if banks{1}(1,2) < banks{2}(1,2)
        banksout{1} = banks{1};
        banksout{2} = banks{2};
    else
        banksout{1} = banks{2};
        banksout{2} = banks{1};
    end
elseif es(1) == 'W'
    if banks{1}(1,2) > banks{2}(1,2)
        banksout{1} = banks{1};
        banksout{2} = banks{2};
    else
        banksout{1} = banks{2};
        banksout{2} = banks{1};
    end
end    
    
% Plotting
if plotornot
    figure
    imshow(Io); hold on
    plot(banksout{1}(:,1),banksout{1}(:,2),'m','linewidth',2); 
    plot(banksout{2}(:,1),banksout{2}(:,2),'m','linewidth',2); 
end

end % banklines_from_mask function

