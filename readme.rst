Description
===========

TBilateral is a spatial smoothing filter that uses the bilateral
filtering algorithm. It does a nice job of smoothing while retaining
picture structure.

This is a port of the Avisynth plugin TBilateral by tritical, version
0.9.11.


Usage
=====
::

    tbilateral.TBilateral(clip clip, [clip ppclip=None, int[] diameter=5, float[] sdev=1.4, float[] idev=7.0, float[] cs=1.0, bint d2=False, int kerns=2, int kerni=2, int restype=0, int[] planes=<all>])


Parameters:
    *clip*
        Clip to denoise.

        Must have constant format and dimensions, 8..16 bits per
        sample, and integer sample type. The dimensions must be
        multiples of 2.

    *ppclip*
        Specifies a pre-filtered clip for TBilateral to take pixel
        values from when doing the luminance difference calculations.

        The general recommendation for pre-processing is a gaussian
        blur with standard deviation equal to the *sdev* settings being
        used. Using a prefiltered clip should help in removing impulse
        noise (i.e. outliers with very large pixel differences) which
        standard bilateral filtering will not touch. It does tend to
        produce artifacts sometimes, especially around very fine
        details. Another recommendation for pre-processing is a
        center-weighted median or adaptive median.

        Must have the same format, dimensions, and number of frames as
        *clip*.

        Default: None.

    *diameter*
        This sets the size of the diameter of the filtering window.
        Larger values mean more pixels will be included in the average,
        but are also slower.
        
        Must be an odd number greater than 1. This must be less than
        the width of the video and less than the height of the video.

        Default: 5 for the first plane, and the previous plane's
        *diameter* for the other planes.

    *sdev*
        This sets the spatial deviations. The larger *sdev* is, the
        less effect distance will have in the weighting of pixels in
        the average. That is, as you increase *sdev*, distant pixels
        will have more weight. This must be greater than 0.

        To get a better idea of what this setting does try setting
        *idev* to high values, and then gradually increase *sdev* from
        0 on up while keeping *idev* constant.
        
        Increasing this setting will increase the strength of the
        smoothing.

        Default: 1.4 for the first plane, and the previous plane's
        *sdev* for the other planes.

    *idev*
        This sets the pixel intensity deviations (or color deviations
        in the case of the chroma planes). The larger *idev* is, the
        less effect pixel difference will have in the weighting of
        pixels in the average. That is, as you increase *idev*, pixels
        that are very different from the current pixel will have more
        weight. This must be greater than 0.

        Try increasing this setting while keeping *sdev* constant to
        get a better idea of what this does. Increasing this setting
        will increase the strength of the smoothing.

        Default: 7.0 for the first plane, and the previous plane's
        *idev* for the other planes.

    *cs*
        This value is multiplied to the center pixel's spatial weight
        value. A value of 1 does nothing, less than 1 means the center
        pixel will have less weight than normal, greater than 1 means
        the center pixel will have more weight than normal, 0 gives the
        center pixel no weight. This must be at least 0. Setting *cs*
        to 0 will give you SUSAN denoising.

        Default: 1.0 for the first plane, and the previous plane's *cs*
        for the other planes.

    *d2*
        This setting makes TBilateral use the second derivative instead
        of the first when doing the intensity calculations. Using *d2*
        should give better results on smooth gradients or anything that
        fits the piecewise linear model. Setting *d2* to False will
        give better results on images that have uniformly colored areas
        with sharp edges (anything that fits the piecewise constant
        model). The actual difference between the two is usually not
        big for most sources. The effect is rather subtle.
        
        Default: False.

    *kerns*
        This specifies what kernel is used for the domain weights. The
        possible choices are:

        * 0 - Andrews' wave
        * 1 - El Fallah Ford
        * 2 - Gaussian
        * 3 - Huber’s mini-max
        * 4 - Lorentzian
        * 5 - Tukey bi-weight
        * 6 - Linear descent
        * 7 - Cosine
        * 8 - Flat
        * 9 - Inverse

        See the following paper for a description of all the kernels
        and their properties:
        http://dsp7.ee.uct.ac.za/~jfrancis/publicationsDir/PRASA2003.pdf
        (dead link). Possible alternative:
        http://www.prasa.org/proceedings/2003/prasa03-09.pdf

        Default: 2 (Gaussian).

    *kerni*
        This specifies what kernel is used for the range weights. The
        possible choices are the same as for *kerns*.

        Default: 2 (Gaussian).

    *restype*
        This specifies how the weights and pixel values are combined to
        obtain the final result. Possible options:

        * 0 - Mean (weighted average)
        * 1 - Median (weighted median)
        * 2 - CW-Median (weighted median + extra center pixel weight)
        * 3 - Multiple linear regression (best fit plane)

        Default: 0 (Mean).

    *planes*
        Select which planes to process. Any unprocessed planes will be
        copied from *clip*.

        Default: all of them.


Compilation
===========

::

    meson build && cd build
    ninja


License
=======

GPL v2, like the Avisynth plugin.
