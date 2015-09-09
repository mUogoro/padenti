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

#include <cstdlib>
#include <iostream>

#include <padenti/cv_image_loader.hpp>
#include <padenti/uniform_image_sampler.hpp>
#include <padenti/training_set.hpp>
#include <padenti/tree.hpp>
#include <padenti/cl_tree_trainer.hpp>


#define TREE_DEPTH (20)
#define TRAIN_DEPTH (20)
#define N_SAMPLES (2048)
#define N_FEATURES (2048)
#define N_THRESHOLDS (20)

#define USE_CPU (false)

static const unsigned char RGB2LABEL[][3] ={
  {255, 0, 0},      // hand
  {0, 0, 255},    // body
};
static const size_t N_LABELS = sizeof(RGB2LABEL)/(sizeof(unsigned char)*3);


typedef CVImageLoader<unsigned short, 1> DepthmapLoaderT;
typedef CVRGBLabelsLoader LabelsLoaderT;
typedef UniformImageSampler<unsigned short, 1> SamplerT;
typedef TrainingSet<unsigned short, 1> TrainingSetT;
typedef Tree<short int, 2, N_LABELS> TreeT;
typedef TreeTrainerParameters<short int, 2> TreeTrainerParametersT;
typedef CLTreeTrainer<unsigned short, 1, short int, 2, N_LABELS> TreeTrainerT;


int main(int argc, const char *argv[])
{
  if (argc==3)
  {
    DepthmapLoaderT depthmapLoader;
    LabelsLoaderT labelsLoader(RGB2LABEL, N_LABELS);
    SamplerT sampler(N_SAMPLES, atoi(argv[1]));
    TrainingSetT trainingSet(argv[2], "_depth.png", "_labels.png", N_LABELS,
			     depthmapLoader, labelsLoader, sampler);

    TreeT tree(atoi(argv[1]), TREE_DEPTH);
    TreeTrainerT trainer(".", USE_CPU);

    TreeTrainerParametersT params;
    params.nFeatures = N_FEATURES;
    params.nThresholds = N_THRESHOLDS;
    params.computeFRange = false;
    params.featLowBounds[0] = -60;
    params.featLowBounds[1] = -60;
    params.featUpBounds[0] = 60;
    params.featUpBounds[1] = 60;
    params.thrLowBound = -200;
    params.thrUpBound = 200;
    params.randomThrSampling = true;
    params.perLeafSamplesThr = static_cast<float>(N_SAMPLES)/N_LABELS;
    //params.perLeafSamplesThr = 1;

    try
    {
      trainer.train(tree, trainingSet, params, 1, TRAIN_DEPTH);
      
      std::stringstream treeName;
      treeName << "tree" << argv[1] << ".xml";
      tree.save(treeName.str());
    }
    catch (cl::Error err)
    {
      std::cerr << err.what() << ": " << err.err() << std::endl;
    }


    return 0;
  }

  return 1;
}
