# Padenti:  An OpenCL-accelerated Random Forests implementation for Computer Vision applications using local features

Padenti is an Open Source implementation of the Random Forests classifier specifically
suited for Computer Vision applications that use simple per-pixel local features (e.g.
class labeling, objects segmentation etc.). Both the training and the prediction are
accelerated on GPUs using the OpenCL framework.

The library has been developed by the
[Engineering for Health and Wellbeing group](http://www.ehw.ieiit.cnr.it/?q=computervision) at the
[Institute of Electronics, Computer and Telecommunication Engineering](http://www.ieiit.cnr.it)
of the National Research Council of Italy (CNR) and is distributed under the [LGPLv3 license](https://www.gnu.org/licenses/lgpl-3.0.en.html).

Features include:
- fast training of large datasets using OpenCL
- support for both NVIDIA and AMD GPUs
- support of arbitrary image pixel type and number of channels
- support of arbitrary per-pixel features through a custom OpenCL C function
  
"Padenti" stands for "Forest" in Sardinian language (in its variant of the Mogoro village).


## Dependencies and installation
### Dependencies
- A GNU Linux or Windows system
- An OpenCL environment from either Nvidia or AMD
- A CPU with SS2 instructions support
- OpenCV (components highgui and imgproc)
- Boost (components random, filesystem, chrono and log)
- cmake 3.1 (for compilation and installation)
- Doxygen (for documentation generation)
- [pthread-win32](https://www.sourceware.org/pthreads-win32/) (Windows only)
- [cygwin](https://www.cygwin.com/) (xxd Windows port needed for compilation).

### Compilation and installation
On Linux
```
git clone https://github.com/mUogoro/padenti.git padenti
cd padenti
mkdir build && cd build
cmake ..
make && make doc && make install
```
On Windows, open the generated solution file in Visual Studio.

## Citing
If you use Padenti in a scientific publication, please cite the following article

Daniele Pianu, Roberto Nerino, Claudia Ferraris, and Antonio Chimienti 
**A novel approach to train random forests on GPU for computer vision applications using local features**
*International Journal of High Performance Computing Applications*
December 29, 2015 doi:10.1177/1094342015622672
[abstract](http://hpc.sagepub.com/content/early/2015/12/29/1094342015622672.abstract) [bib](http://hpc.sagepub.com/citmgr?type=bibtex&gca=sphpc%3B1094342015622672v1)


For more information about library usage and a small tutorial please consult the
[Doxygen documentation](http://muogoro.github.io/padenti).
