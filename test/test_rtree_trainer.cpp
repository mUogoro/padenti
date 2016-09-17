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
#include <padenti/training_set_r.hpp>
#include <padenti/rtree.hpp>
#include <padenti/cl_rtree_trainer.hpp>



#define TREE_DEPTH (15)
#define TRAIN_DEPTH (TREE_DEPTH)
#define N_SAMPLES (512)
#define N_FEATURES (512)
#define N_THRESHOLDS (8)
#define MIN_LEAF_SAMPLES (50)


#define USE_CPU (false)


typedef CVImageLoader<unsigned short, 1> DepthmapLoaderT;
typedef UniformImageSampler<unsigned short, 1> SamplerT;
typedef TrainingSetR<unsigned short, 1, 4> TrainingSetT;
typedef RTree<short int, 2, 4> TreeT;
typedef TreeTrainerParameters<short int, 2> TreeTrainerParametersT;
typedef CLRTreeTrainer<unsigned short, 1, short int, 2, 4> TreeTrainerT;


int main(int argc, const char *argv[])
{
  if (argc==4)
  {
    // Load the depthmap paths and corresponding rotation, expressed as euler angles
    /*************************************************************************************/
    // Note: current implementation work only with regression values in R4, hence set    */
    /* forth component of the image regression value to zero                             */
    /*************************************************************************************/
    std::vector<std::string> dataPaths;
    std::vector<std::vector<float> > values;
    
    std::ifstream dataPathsFile(argv[2]);
    std::ifstream valuesFile(argv[3]);
    
    std::string line;
    while (getline(dataPathsFile, line))
    { 
      dataPaths.push_back(line);
    }
    while (getline(valuesFile, line))
    {
      std::stringstream ssline(line);
      std::vector<float> value(4);
      ssline >> value.at(0); ssline >> value.at(1); ssline >> value.at(2);
      value.at(0) *= M_PI/180.f;
      value.at(1) *= M_PI/180.f;
      value.at(2) *= M_PI/180.f;
      value.at(3) = 0;
      values.push_back(value);
    }


    DepthmapLoaderT depthmapLoader;
    SamplerT sampler(N_SAMPLES, atoi(argv[1]));
    TrainingSetT trainingSet(dataPaths, depthmapLoader, values, sampler);

    TreeT tree(atoi(argv[1]), TREE_DEPTH);
    TreeTrainerT trainer(".", USE_CPU);

    TreeTrainerParametersT params;
    params.nFeatures = N_FEATURES;
    params.nThresholds = N_THRESHOLDS;
    params.computeFRange = false;
    params.featLowBounds[0] = -68;
    params.featLowBounds[1] = -68;
    params.featUpBounds[0] = 68;
    params.featUpBounds[1] = 68;
    params.thrLowBound = -184;
    params.thrUpBound = 184;
    params.randomThrSampling = true;
    params.perLeafSamplesThr = MIN_LEAF_SAMPLES;
    //params.perLeafSamplesThr = 1;

    //try
    //{
      trainer.train(tree, trainingSet, params, 1, TRAIN_DEPTH);
      
      std::stringstream treeName;
      treeName << "tree" << argv[1] << ".xml";
      tree.save(treeName.str());
    //}
    //catch (cl::Error err)
    //{
    //  std::cerr << err.what() << ": " << err.err() << std::endl;
    //}


    return 0;
  }

  return 1;
}
