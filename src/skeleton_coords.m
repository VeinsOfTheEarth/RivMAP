function xy = skeleton_coords(Iskel, xst, yst, es)

% Traces an input skeleton image to find the coordinates. Similar to
% bwtraceboundaries except only one pass is made along the length of the
% skeleton (bwboundaries performs a "there and back" search). The
% coordinates are then put into upstream-to-downstream orientation. The
% input skeleton should be completely pruned, else the trace may halt
% prematurely.

% INPUTS:      Iskel - image of branchless skeleton to be traced
%                xst - x-coordinate of trace starting point
%                yst - y-coordinate of trace starting point
%                 es - "exit sides", a two-letter string array from NESW 
%                       options (e.g. 'NS') where the first letter represents 
%                       the side of the image that the downstream end of the 
%                       river enters the image and the second letter represents
%                       the side of the image where the river exits.
%
% OUTPUTS:        xy - Nx2 array of x,y coordinates of skeleton trace,
%                      arranged in upstream->downstream order, where N is
%                      the number of pixels in the skeleton

x = xst;
y = yst;
c = 1; 
[nr, nc] = size(Iskel);

while 1
    if y(c)-1 > 0 && Iskel(y(c)-1,x(c)) == 1  % N
        x(c+1) = x(c);
        y(c+1) = y(c) - 1;
        Iskel(y(c),x(c)) = 0;
    elseif x(c) + 1 <= nc && Iskel(y(c),x(c)+1) == 1  % E
        x(c+1) = x(c) + 1;
        y(c+1) = y(c);
        Iskel(y(c),x(c)) = 0;
    elseif y(c) + 1 <= nr && Iskel(y(c)+1,x(c)) == 1  % S
        x(c+1) = x(c);
        y(c+1) = y(c) + 1;
        Iskel(y(c),x(c)) = 0;
    elseif x(c) - 1 > 0 && Iskel(y(c),x(c)-1) == 1  % W
        x(c+1) = x(c) - 1;
        y(c+1) = y(c);
        Iskel(y(c),x(c)) = 0;
    elseif y(c)-1 > 0 && x(c) + 1 <= nc && Iskel(y(c)-1,x(c)+1) == 1  % NE
        x(c+1) = x(c) + 1;
        y(c+1) = y(c) - 1;
        Iskel(y(c),x(c)) = 0;
    elseif y(c) + 1 <= nr && x(c) + 1 <= nc && Iskel(y(c)+1,x(c)+1) == 1  % SE
        x(c+1) = x(c) + 1;
        y(c+1) = y(c) + 1;
        Iskel(y(c),x(c)) = 0;
    elseif y(c) + 1 <= nr && x(c) - 1 > 0 && Iskel(y(c)+1,x(c)-1) == 1  % SW
        x(c+1) = x(c) - 1;
        y(c+1) = y(c) + 1;
        Iskel(y(c),x(c)) = 0;
    elseif y(c)-1 > 0 && x(c) - 1 > 0 && Iskel(y(c)-1,x(c)-1) == 1  % NW
        x(c+1) = x(c) - 1;
        y(c+1) = y(c) - 1;
        Iskel(y(c),x(c)) = 0;
    elseif c > length(Iskel)*3
        break
    else
        break
    end
    c = c + 1;
end 
xy = [x; y]';

% Order coordinates upstream->downstream
if es(1) == 'W'
    if xy(1,1) > xy(end,1)
        xy = flipud(xy);
    end
elseif es(1) == 'N'
    if xy(1,2) > xy(end,2)
        xy = flipud(xy);
    end
elseif es(1) == 'E'
    if xy(1,1) < xy(end,1)
        xy = flipud(xy);
    end
elseif es(1) == 'S'
    if xy(1,2) < xy(end,2)
        xy = flipud(xy);
    end
end
