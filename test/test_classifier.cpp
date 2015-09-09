/******************************************************************************
 * Padenti Library
 *
 * Copyright (C) 2015  Daniele Pianu <daniele.pianu@ieiit.cnr.it>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 ******************************************************************************/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <padenti/image.hpp>
#include <padenti/tree.hpp>
#include <padenti/cl_classifier.hpp>
#include <padenti/cv_image_loader.hpp>


static const unsigned char RGB2LABEL[][3] ={
  {255, 0, 0},      // hand
  {0, 0, 255}     // body
};
static const size_t N_LABELS = sizeof(RGB2LABEL)/(sizeof(unsigned char)*3);

typedef Tree<short int, 2, N_LABELS> TreeT;
typedef Image<unsigned short, 1> DepthT;
typedef Image<unsigned char, 1> MaskT;
typedef Image<float, N_LABELS> PredictionT;
typedef CVImageLoader<unsigned short, 1> ImageLoaderT;
typedef CLClassifier<unsigned short, 1, short, 2, N_LABELS> ClassifierT;


#define USE_CPU (false)

int main(int argc, char *argv[])
{
  // Init the classifier
  const int nTrees = argc-1;
  TreeT trees[nTrees];
  ClassifierT classifier("kernels", USE_CPU);

  // Load the trees
  for (int t=0; t<argc-2; t++)
  {
    trees[t].load(argv[t+2], t);
    classifier << trees[t];
  }

  cv::namedWindow("hand");
  cv::namedWindow("background");

  // Load depth
  ImageLoaderT imageLoader;
  DepthT depthmap = imageLoader.load(argv[1]);
  MaskT mask(depthmap.getWidth(), depthmap.getHeight());

  // TODO: explain masking
  cv::Mat cvDepth(depthmap.getHeight(), depthmap.getWidth(), CV_16U,
		  reinterpret_cast<unsigned char*>(depthmap.getData()));
  cv::Mat cvMask(cvDepth.rows, cvDepth.cols, CV_8U,
		 reinterpret_cast<unsigned char*>(mask.getData()));
  cvMask.setTo(0);
  cvMask.setTo(1, cvDepth>0);

  // Prediction
  PredictionT prediction(cvDepth.cols, cvDepth.rows);
  classifier.predict(depthmap, prediction, mask);
  
  // Show prediction result
  cv::Mat cvHand(cvDepth.rows, cvDepth.cols, CV_32F,
                 reinterpret_cast<void*>(prediction.getData()));
  cv::Mat cvBackground(cvDepth.rows, cvDepth.cols, CV_32F,
                       reinterpret_cast<void*>(prediction.getData()+cvDepth.rows*cvDepth.cols));

  cv::imshow("hand", cvHand);
  cv::imshow("background", cvBackground);

  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}
