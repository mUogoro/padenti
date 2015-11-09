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

function [] = test_padenti_train(trainingSetPath, ...
                                 imagesSuffix, labelsSuffix)
% TEST_PADENTI_TRAIN Test Padenti library wrapper for training

  % Labels math for the MSRC_V2 dataset
  RGB2LABEL = uint8([...
    128, 0, 0;     % building 
    0, 128, 0;     % grass 
    128, 128, 0;   % tree 
    0, 0, 128;     % cow 
    0, 128, 128;   % sheep
    128, 128, 128; % sky
    192, 0, 0;     % aeroplane
    64, 128, 0;    % water
    192, 128, 0;   % face
    64, 0, 128;    % car
    192, 0, 128;   % bicycle
    64, 128, 128;  % flower
    192, 128, 128; % sign
    0, 64, 0;      % bird
    128, 64, 0;    % book
    0, 192, 0;     % chair
    128, 64, 128;  % road
    0, 192, 128;   % cat
    128, 192, 128; % dog
    64, 64, 0;     % body
    192, 64, 0;    % boat
]);
  N_LABELS = size(RGB2LABEL,1);
  N_CHANNELS = 4; % Note: N_CHANNELS should be 3 since we work with RGB
                  % images, but since OpenCL supports only 4-channel
                  % 32bit images we need to add an additional empty channel

  imgFiles = dir(strcat(trainingSetPath, '/*', imagesSuffix));
  
  nImages = numel(imgFiles);
  
  % Allocate a cell array for images and labels
  images = cell(1, nImages);
  labels = cell(1, nImages);
  
  % Read images and labels
  % Note: images storing is different in matlab (consecutive image
  % channels) w.r.t. OpenCL 4-channel images (consecutive triplets),
  % thus apply permutations to matlab images to store triplets first
  % when working with N_CHANNELS <=4, otherwise left images unchanged
  % (Padenti treats images with more than 4 channels as layered images,
  % the same way as matlab does)
  for n=1:numel(imgFiles)
     
      imgName = imgFiles(n).name;
      imgData = imread(strcat(trainingSetPath,'/',imgName), 'png');
      imgSize = size(imgData);
      
      % Compute integral image
      intImageR = uint32(integralImage(imgData(:,:,1)));
      intImageG = uint32(integralImage(imgData(:,:,2)));
      intImageB = uint32(integralImage(imgData(:,:,3)));
      
      intImage = zeros(imgSize(1),imgSize(2),N_CHANNELS, 'uint32');
      intImage(:,:,1) = intImageR(2:end,2:end);
      intImage(:,:,2) = intImageG(2:end,2:end);
      intImage(:,:,3) = intImageB(2:end,2:end);
      if N_CHANNELS <= 4
        images{1,n} = permute(intImage, [3,1,2]);
      else
        images{1,n} = intImage;
      end
      
      prefixSize = size(imgName,2)-size(imagesSuffix,2);
      labelsImgName = strcat(imgName(:,1:prefixSize), labelsSuffix);
      rgbLabels = ...
          imread(strcat(trainingSetPath, '/', labelsImgName), 'png');
      
      % Convert RGB labels value to corresponding labels index
      % using the RGB2LABEL map
      perImgLabels = zeros(imgSize(1), imgSize(2), 'uint8');
      for l=1:N_LABELS
          mask = false(imgSize(1), imgSize(2));
          mask(1:imgSize(1),1:imgSize(2)) = ...
              rgbLabels(:,:,1)==RGB2LABEL(l,1) & ...
              rgbLabels(:,:,2)==RGB2LABEL(l,2) & ...
              rgbLabels(:,:,3)==RGB2LABEL(l,3);
          perImgLabels(mask) = l;
      end
      if N_CHANNELS<=4
        labels{1,n} = permute(perImgLabels, [3,1,2]);
      else
        labels{1,n} = perImgLabels;
      end
      
  end
  
  % Done with training set loading, start training:
  % - init training parameters
  trainParams = struct('nTrees', 5, ...
                       'depth', 20, ...
                       'nSamples', 512, ...
                       'nFeatures', 512, ...
                       'nThresholds', 20, ...
                       'featLowBounds', [-60 -60 1 1 0,  ...
                                         -60 -60 1 1 0], ...
                       'featUpBounds', [60 60 13 13 3,  ...
                                        60 60 13 13 3], ...
                       'thrLowBound', -160, ...
                       'thrUpBound', 160, ...
                       'perLeafSamplesThr', 38);
  
  padenti_train(trainParams, images, labels, N_LABELS);
  
  
end

