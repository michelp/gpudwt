/// 
/// @file    dwt.h
/// @author  Martin Jirman (207962@mail.muni.cz)
/// @brief   Entry points for CUDA implementaion of 9/7 and 5/3 DWT.
/// @date    2011-01-20 11:41
///
///
/// Following conditions are common for all four DWT functions:
/// - Both input and output images are stored in GPU memory with no padding
///   of lines or interleaving of pixels.
/// - DWT coefficients are stored as follows: Each band is saved as one
///   consecutive chunk (no padding/stride/interleaving). Deepest level bands
///   (smallest ones) are stored first (at the beginning of the input/output
///   buffers), less deep bands follow. There is no padding between stored
///   bands in the buffer. Order of bands of the same level in the buffer is
///   following: Low-low band (or deeper level subbands) is stored first.
///   Vertical-low/horizontal-high band follows. Vertical-high/horizonal-low
///   band is saved next and finally, the high-high band is saved. Out of all
///   low-low bands, only th edeepest one is saved (right at the beginning of
///   the buffer), others are replaced with deeper level subbands.
/// - Input images of all functions won't be preserved (will be overwritten).
/// - Input and output buffers can't overlap.
/// - Size of output buffer must be greater or equal to size of input buffer.
///
/// There are no common compile time settings (buffer size, etc...) for
/// all DWTs, because each DTW type needs different amount of GPU resources.
/// Instead, each DWT type has its own compile time settings, which can be
/// found in *.cu file, where it is implemented.
///

#ifndef DWT_CUDA_H
#define	DWT_CUDA_H


namespace dwt_cuda {
  
  
  /// Forward 5/3 2D DWT. See common rules (above) for more details.
  /// @param in      Expected to be normalized into range [-128, 127].
  ///                Will not be preserved (will be overwritten).
  /// @param out     output buffer on GPU
  /// @param sizeX   width of input image (in pixels)
  /// @param sizeY   height of input image (in pixels)
  /// @param levels  number of recursive DWT levels
  void fdwt53(int * in, int * out, int sizeX, int sizeY, int levels);
  
  
  /// Reverse 5/3 2D DWT. See common rules (above) for more details.
  /// @param in      Input DWT coefficients. Format described in common rules.
  ///                Will not be preserved (will be overwritten).
  /// @param out     output buffer on GPU - will contain original image
  ///                in normalized range [-128, 127].
  /// @param sizeX   width of input image (in pixels)
  /// @param sizeY   height of input image (in pixels)
  /// @param levels  number of recursive DWT levels
  void rdwt53(int * in, int * out, int sizeX, int sizeY, int levels);
  
  
  /// Forward 9/7 2D DWT. See common rules (above) for more details.
  /// @param in      Input DWT coefficients. Should be normalized (in range 
  ///                [-0.5, 0.5]). Will not be preserved (will be overwritten).
  /// @param out     output buffer on GPU - format specified in common rules
  /// @param sizeX   width of input image (in pixels)
  /// @param sizeY   height of input image (in pixels)
  /// @param levels  number of recursive DWT levels
  void fdwt97(float * in, float * out, int sizeX, int sizeY, int levels);
  
  
  /// Reverse 9/7 2D DWT. See common rules (above) for more details.
  /// @param in      Input DWT coefficients. Format described in common rules.
  ///                Will not be preserved (will be overwritten).
  /// @param out     output buffer on GPU - will contain original image
  ///                in normalized range [-0.5, 0.5].
  /// @param sizeX   width of input image (in pixels)
  /// @param sizeY   height of input image (in pixels)
  /// @param levels  number of recursive DWT levels
  void rdwt97(float * in, float * out, int sizeX, int sizeY, int levels);
  
  
} // namespace dwt_cuda



#endif	// DWT_CUDA_H

