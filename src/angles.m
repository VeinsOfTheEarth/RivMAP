function A = angles(xy)

% Computes angles between nodes along an input line defined by x,y
% coordinates.

% INPUTS:      xy - Nx2 vector of coordinates of line to compute angles
%                   along
%              
%
% OUTPUTS:     A  - Nx1 vector of angles along the xy line, in radians. The
%                   first angle is NaN.

% Work with column vectors
if size(xy,1) < size(xy,2)
    xy = xy';
    rotated = 1;
else
    rotated = 0;
end

X0 = xy(1:end-1,1);
X1 = xy(2:end,1);

Y0 = xy(1:end-1,2);
Y1 = xy(2:end,2);

A = atan2(Y1-Y0,X1-X0);
A = unwrap(A); % shift to avoid jumps in signal

% First angle is NaN
A = [NaN; A];

if rotated
    A = A';
end


