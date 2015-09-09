#include <training_set_image.hpp>
#include <cv_image_loader.hpp>
#include <uniform_image_sampler.hpp>
#include <training_set.hpp>

int main(int argc, const char *argv[])
{
  if (argc==2)
  {
    unsigned char rgb2labels[][3] = {{255, 0, 0},      /* thumb 3 */
				     {204, 51, 51},    /* thumb 2 */
				     {204, 102, 102},  /* thumb 1 */
				     {255, 255, 0},    /* pinky 3 */
				     {255, 255, 51},   /* pinky 2 */
				     {255, 255, 102},  /* pinky 1 */
				     {0, 255, 0},      /* ring 3 */
				     {51, 255, 51},    /* ring 2 */
				     {102, 255, 102},  /* ring 1 */
				     {0, 0, 255},     /* middle 3 */
				     {51, 51, 255},   /* middle 2 */
				     {102, 102, 255}, /* middle 1 */
				     {255, 0, 255},   /* index 3 */
				     {255, 51, 255},  /* index 2 */
				     {255, 102, 255}, /* index 1 */
				     {255, 25, 25},   /* thumb palm */
				     {255, 255, 153}, /* pinky palm */
				     {153, 255, 153}, /* ring palm */
				     {153, 153, 255}, /* middle palm */
				     {255, 153, 255}, /* index palm */
				     {255, 76, 0},    /* palm */
				     {76, 25, 0}};    /* wrist */
    unsigned int nClasses = sizeof(rgb2labels)/(sizeof(unsigned char)*3);

    CVImageLoader<unsigned short, 1> imgLoader;
    CVRGBLabelsLoader labelsLoader(rgb2labels, nClasses);
    UniformImageSampler<unsigned short, 1> sampler(2048);

    TrainingSet<unsigned short, 1> trainingSet(argv[1], "_depth.png", ".png", nClasses,
					       imgLoader, labelsLoader, sampler);

    return 0;
  }

  return 1;
}
