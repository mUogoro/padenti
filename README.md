# Padenti:  An OpenCL-accelerated Random Forests implementation for Computer Vision applications using local features

Padenti is an Open Source implementation of the Random Forests classifier specifically
suited for Computer Vision applications that use simple per-pixel local features (e.g.
class labeling, objects segmentation etc.). Both the training and the prediction are
accelerated on GPUs using the OpenCL framework.

The library has been developed by the
[Engineering for Health and Wellbeing group](http://www.ehw.ieiit.cnr.it/?q=computervision) at the
[Institute of Electronics, Computer and Telecommunication Engineering](http://www.ieiit.cnr.it)
of the National Research Council of Italy (CNR).

Features include:
- fast training of large datasets using OpenCL
- support for both NVIDIA and AMD GPUs
- support of arbitrary image pixel type and number of channels
- support of arbitrary per-pixel features through a custom OpenCL C function
  
"Padenti" is the word in Sardinian language (in its variant of the Mogoro village) for "Forest".

For installation, usage and a small tutorial please consult the Doxygen documentation.
