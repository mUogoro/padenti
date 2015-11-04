%******************************************************************************
% Padenti Library
%
% Copyright (C) 2015  Daniele Pianu <daniele.pianu@ieiit.cnr.it>
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU Lesser General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Lesser General Public License for more details.
%
% You should have received a copy of the GNU Lesser General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>
%******************************************************************************/

function [ prediction ] = test_padenti_predict( instance, image )
%TEST_PADENTI_PREDICT Test Padenti library wrapper for prediction

  % Work with integer image with 4 channels
  aggImg = zeros(size(image,1), size(image,2), 4,'uint32');

  % Compute integral image
  % Note: images storing is different in matlab (consecutive image
  % channels) w.r.t. OpenCL 4-channel images (consecutive triplets),
  % thus apply permutations to matlab images to store triplets first
  intImageR = uint32(integralImage(image(:,:,1)));
  intImageG = uint32(integralImage(image(:,:,2)));
  intImageB = uint32(integralImage(image(:,:,3)));
  aggImg(:,:,1) = intImageR(2:end,2:end);
  aggImg(:,:,2) = intImageG(2:end,2:end);
  aggImg(:,:,3) = intImageB(2:end,2:end);
  intImg = permute(aggImg, [3,1,2]);
  
  prediction = padenti_predict(instance, intImg);
  
end

