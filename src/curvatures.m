function C = curvatures(xy)

% Computes angles between nodes along an input line defined by x,y
% coordinates using a central-differencing scheme.

% INPUTS:      xy - Nx2 vector of coordinates of line to compute angles
%                   along
%              
%
% OUTPUTS:     C  - Nx1 vector of curvatures along the xy line in units of
%                   1/u, where u is the unit of the input xy data.

% Along-line distances
s = [0; cumsum(sqrt(diff(xy(:,1)).^2+diff(xy(:,2)).^2))];
% Angles
A = angles(xy);
   
% Central differencing for curvatures
% See http://terpconnect.umd.edu/~toh/spectrum/Differentiation.html
sd = [s(3);s(1:end-1)]; 
su = [s(2:end);s(end-2)];
Ad = [A(3);A(1:end-1)]; 
Au = [A(2:end);A(end-2)];
C = -((Au-A)./(su-s).*(s-sd)+(A-Ad)./(s-sd).*(su-s))./(su-sd);
