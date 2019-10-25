#if defined(_WIN32)
#define _USE_MATH_DEFINES
#endif


#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <VapourSynth.h>
#include <VSHelper.h>


enum KernelTypes {
    AndrewsWave = 0,
    ElFallahFord,
    Gaussian,
    HubersMiniMax,
    Lorentzian,
    TukeyBiWeight,
    LinearDescent,
    Cosine,
    Flat,
    Inverse
};


enum ResTypes {
    Mean = 0,
    Median,
    CWMedian,
    MultipleLinearRegression
};


struct TBilateralData {
    VSNodeRef *clip;
    VSNodeRef *ppclip;
    const VSVideoInfo *vi;

    int process[3];

    int diameter[3];
    double sDev[3];
    double iDev[3];
    double cs[3];
    bool d2;
    int kernS;
    int kernI;
    int resType;

    int pixel_max;

    double *spatialWeights[3];
    double *diffWeights[3];

    void (*process_frame)(const VSFrameRef *src, const VSFrameRef *pp, VSFrameRef *dst, const TBilateralData *d, const VSAPI *vsapi);
};


#define MINS 0.00000000000001


static double kernelValue(double x, double sigma, int kernel) {
    switch (kernel) {
        case AndrewsWave: // Andrews' wave
            if (x <= sigma)
                return ((sin((M_PI * x) / sigma) * sigma) / M_PI);
            return 0.0;
        case ElFallahFord: // El Fallah Ford
            return (1.0 / sqrt(1.0 + ((x * x) / (sigma * sigma))));
        case Gaussian: // Gaussian
            return (exp(-((x * x) / (2.0 * sigma * sigma))));
        case HubersMiniMax: // Huber’s mini-max
            if (x <= sigma)
                return (1.0 / sigma);
            return (1.0 / x);
        case Lorentzian: // Lorentzian
            return (2.0 / (2.0 * sigma * sigma + x * x));
        case TukeyBiWeight: // Tukey bi-weight
            if (x <= sigma)
                return (0.5 * pow((1.0 - ((x * x) / (sigma * sigma))), 2));
            return 0.0;
        case LinearDescent: // Linear descent
            if (x <= sigma)
                return (1.0 - (x / sigma));
            return 0.0;
        case Cosine: // Cosine
            if (x <= sigma)
                return (cos((M_PI * x) / (2.0 * sigma)));
            return 0.0;
        case Flat: // Flat
            if (x <= sigma)
                return (1.0 / sigma);
            return 0.0;
        case Inverse: // Inverse
            if (x <= sigma) {
                if (x != 0.0)
                    return (1.0 / x);
                return 1.0;
            }
            return 0.0;
    }
    return 0.0;
}


static void buildTables(TBilateralData *d) {
    for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
        if (!d->process[plane])
            continue;

        int window = d->diameter[plane] * d->diameter[plane];
        int radius = d->diameter[plane] >> 1;

        d->spatialWeights[plane] = vs_aligned_malloc<double>(window * sizeof(double), 16);
        double *disTable = vs_aligned_malloc<double>(window * sizeof(double), 16);

        for (int b = 0, y = -radius; y <= radius; ++y) {
            int temp = y * y;
            for (int x = -radius; x <= radius; ++x)
                disTable[b++] = sqrt((double)(temp + x * x));
        }

        for (int x = 0; x < window; ++x)
            d->spatialWeights[plane][x] = kernelValue(disTable[x], d->sDev[plane], d->kernS);
        d->spatialWeights[plane][radius * d->diameter[plane] + radius] *= d->cs[plane];

        int diff_size = d->pixel_max * (d->d2 ? 4 : 2) + 1;

        d->diffWeights[plane] = vs_aligned_malloc<double>(diff_size * sizeof(double), 16);

        for (int x = 0; x <= diff_size / 2; ++x)
            d->diffWeights[plane][diff_size / 2 + x] = d->diffWeights[plane][diff_size / 2 - x] = kernelValue(x / (d->d2 ? 2.0 : 1.0), d->iDev[plane], d->kernI);

        vs_aligned_free(disTable);
    }
}


static void freeTables(TBilateralData *d) {
    for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
        if (!d->process[plane])
            continue;

        vs_aligned_free(d->spatialWeights[plane]);
        vs_aligned_free(d->diffWeights[plane]);
    }
}


// Singular Value Decomposition routine
// taken from numerical recipes in C.

static double pythag(double a, double b) {
    double at = fabs(a), bt = fabs(b), ct;
    if (at > bt) {
        ct = bt / at;
        return at * sqrt(1.0 + ct * ct);
    }
    if (bt > 0.0) {
        ct = at / bt;
        return bt * sqrt(1.0 + ct * ct);
    }
    return 0.0;
}


static void svdcmp(double *a, double *w, double *v) {
    int flag, i, its, j, jj, k, l, nm;
    double c, f, h, s, x, y, z;
    double anorm = 0.0, g = 0.0, scale = 0.0;
    double rv1[3];
    for (i = 0; i < 3; i++) {
        l = i + 1;
        rv1[i] = scale * g;
        g = s = scale = 0.0;
        if (i < 3) {
            for (k = i; k < 3; k++)
                scale += fabs(a[k * 3 + i]);
            if (scale) {
                for (k = i; k < 3; k++) {
                    a[k * 3 + i] = a[k * 3 + i] / scale;
                    s += a[k * 3 + i] * a[k * 3 + i];
                }
                f = a[i * 3 + i];
                g = -copysign(sqrt(s), f);
                h = f * g - s;
                a[i * 3 + i] = f - g;
                if (i != 2) {
                    for (j = l; j < 3; j++) {
                        for (s = 0.0, k = i; k < 3; k++)
                            s += a[k * 3 + i] * a[k * 3 + j];
                        f = s / h;
                        for (k = i; k < 3; k++)
                            a[k * 3 + j] += f * a[k * 3 + i];
                    }
                }
                for (k = i; k < 3; k++)
                    a[k * 3 + i] = a[k * 3 + i] * scale;
            }
        }
        w[i] = scale * g;
        g = s = scale = 0.0;
        if (i < 3 && i != 2) {
            for (k = l; k < 3; k++)
                scale += fabs(a[i * 3 + k]);
            if (scale) {
                for (k = l; k < 3; k++) {
                    a[i * 3 + k] = a[i * 3 + k] / scale;
                    s += a[i * 3 + k] * a[i * 3 + k];
                }
                f = a[i * 3 + l];
                g = -copysign(sqrt(s), f);
                h = f * g - s;
                a[i * 3 + l] = f - g;
                for (k = l; k < 3; k++)
                    rv1[k] = a[i * 3 + k] / h;
                if (i != 2) {
                    for (j = l; j < 3; j++) {
                        for (s = 0.0, k = l; k < 3; k++)
                            s += (a[j * 3 + k] * a[i * 3 + k]);
                        for (k = l; k < 3; k++)
                            a[j * 3 + k] += s * rv1[k];
                    }
                }
                for (k = l; k < 3; k++)
                    a[i * 3 + k] = a[i * 3 + k] * scale;
            }
        }
        anorm = std::max(anorm, (fabs(w[i]) + fabs(rv1[i])));
    }
    for (i = 2; i >= 0; i--) {
        if (i < 2) {
            if (g) {
                for (j = l; j < 3; j++)
                    v[j * 3 + i] = a[i * 3 + j] / a[i * 3 + l] / g;
                for (j = l; j < 3; j++) {
                    for (s = 0.0, k = l; k < 3; k++)
                        s += (a[i * 3 + k] * v[k * 3 + j]);
                    for (k = l; k < 3; k++)
                        v[k * 3 + j] += s * v[k * 3 + i];
                }
            }
            for (j = l; j < 3; j++)
                v[i * 3 + j] = v[j * 3 + i] = 0.0;
        }
        v[i * 3 + i] = 1.0;
        g = rv1[i];
        l = i;
    }
    for (i = 2; i >= 0; i--) {
        l = i + 1;
        g = w[i];
        if (i < 2)
            for (j = l; j < 3; j++)
                a[i * 3 + j] = 0.0;
        if (g) {
            g = 1.0 / g;
            if (i != 2) {
                for (j = l; j < 3; j++) {
                    for (s = 0.0, k = l; k < 3; k++)
                        s += (a[k * 3 + i] * a[k * 3 + j]);
                    f = (s / a[i * 3 + i]) * g;
                    for (k = i; k < 3; k++)
                        a[k * 3 + j] += f * a[k * 3 + i];
                }
            }
            for (j = i; j < 3; j++)
                a[j * 3 + i] = a[j * 3 + i] * g;
        } else {
            for (j = i; j < 3; j++)
                a[j * 3 + i] = 0.0;
        }
        ++a[i * 3 + i];
    }
    for (k = 2; k >= 0; k--) {
        for (its = 0; its < 30; its++) {
            flag = 1;
            for (l = k; l >= 0; l--) {
                nm = l - 1;
                if (fabs(rv1[l]) + anorm == anorm) {
                    flag = 0;
                    break;
                }
                if (fabs(w[nm]) + anorm == anorm)
                    break;
            }
            if (flag) {
                c = 0.0;
                s = 1.0;
                for (i = l; i <= k; i++) {
                    f = s * rv1[i];
                    if (fabs(f) + anorm != anorm) {
                        g = w[i];
                        h = pythag(f, g);
                        w[i] = h;
                        h = 1.0 / h;
                        c = g * h;
                        s = (-f * h);
                        for (j = 0; j < 3; j++) {
                            y = a[j * 3 + nm];
                            z = a[j * 3 + i];
                            a[j * 3 + nm] = y * c + z * s;
                            a[j * 3 + i] = z * c - y * s;
                        }
                    }
                }
            }
            z = w[k];
            if (l == k) {
                if (z < 0.0) {
                    w[k] = -z;
                    for (j = 0; j < 3; j++)
                        v[j * 3 + k] = -v[j * 3 + k];
                }
                break;
            }
            x = w[l];
            nm = k - 1;
            y = w[nm];
            g = rv1[nm];
            h = rv1[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = pythag(f, 1.0);
            f = ((x - z) * (x + z) + h * ((y / (f + copysign(g, f))) - h)) / x;
            c = s = 1.0;
            for (j = l; j <= nm; j++) {
                i = j + 1;
                g = rv1[i];
                y = w[i];
                h = s * g;
                g = c * g;
                z = pythag(f, h);
                rv1[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y = y * c;
                for (jj = 0; jj < 3; jj++) {
                    x = v[jj * 3 + j];
                    z = v[jj * 3 + i];
                    v[jj * 3 + j] = x * c + z * s;
                    v[jj * 3 + i] = z * c - x * s;
                }
                z = pythag(f, h);
                w[j] = z;
                if (z) {
                    z = 1.0 / z;
                    c = f * z;
                    s = h * z;
                }
                f = (c * g) + (s * y);
                x = (c * y) - (s * g);
                for (jj = 0; jj < 3; jj++) {
                    y = a[jj * 3 + j];
                    z = a[jj * 3 + i];
                    a[jj * 3 + j] = y * c + z * s;
                    a[jj * 3 + i] = z * c - y * s;
                }
            }
            rv1[l] = 0.0;
            rv1[k] = f;
            w[k] = x;
        }
    }
}


static int mlre(double *yi, double *wi, int lw, int lh, int cx, int cy, int radius, int diameter) {
    wi += cy * diameter;
    yi += cy * diameter;

    const int la = lw * lh;
    const int lax2 = la * 2;
    const int la2 = la * la;

    double *xlr = (double *)malloc(la * 3 * sizeof(double));
    double *wlr = (double *)malloc(la2 * sizeof(double));
    double *ylr = (double *)malloc(la * sizeof(double));
    double *wxlr = (double *)malloc(la * 3 * sizeof(double));
    double *xtlr = (double *)malloc(la * 3 * sizeof(double));
    double *wylr = (double *)malloc(la * sizeof(double));
    double xtwx[9], xtwxi[9], xtwy[3], blr[3], wjlr[3], vlr[9];

    // compute w and y matrices
    int d = 0, h = 0;
    memset(wlr, 0, la2 * sizeof(double));
    for (int k = 0; k < lh; ++k) {
        const int kt = k * diameter;
        for (int j = cx; j < lw + cx; ++j, ++h, d += la + 1) {
            wlr[d] = wi[kt + j];
            ylr[h] = yi[kt + j];
        }
    }

    // compute x and x' matrices
    d = 0;
    for (int j = 0; j < lh; ++j) {
        const int jt = j * lw * 3;
        for (int k = 0; k < lw; ++k, ++d) {
            xlr[jt + k * 3 + 0] = xtlr[d] = 1;
            xlr[jt + k * 3 + 1] = xtlr[d + la] = j;
            xlr[jt + k * 3 + 2] = xtlr[d + lax2] = k;
        }
    }

    // compute w*x matrix
    for (int j = 0; j < la; ++j) {
        const int j3 = j * 3;
        const int jl = j * la;
        for (int k = 0; k < 3; ++k) {
            wxlr[j3 + k] = 0.0;
            for (int l = 0; l < la; ++l)
                wxlr[j3 + k] += wlr[jl + l] * xlr[l * 3 + k];
        }
    }

    // compute xt*wx matrix
    for (int j = 0; j < 3; ++j) {
        const int j3 = j * 3;
        const int jl = j * la;
        for (int k = 0; k < 3; ++k) {
            xtwx[j3 + k] = 0.0;
            for (int l = 0; l < la; ++l)
                xtwx[j3 + k] += xtlr[jl + l] * wxlr[l * 3 + k];
        }
    }

    // compute svd of xtwx = U*WJ*V'
    svdcmp(xtwx, wjlr, vlr);

    // compute wj inverse + zero small wj's
    for (int i = 0; i < 3; ++i) {
        if (fabs(wjlr[i]) <= FLT_EPSILON)
            wjlr[i] = 0;
        else
            wjlr[i] = 1.0 / wjlr[i];
    }

    // compute wj^-1 * u'
    for (int j = 0; j < 3; ++j) {
        const int j3 = j * 3;
        for (int k = j; k < 3; ++k) {
            double temp = xtwx[j3 + k];
            xtwx[j3 + k] = xtwx[k * 3 + j] * wjlr[j];
            xtwx[k * 3 + j] = temp * wjlr[k];
        }
    }

    // compute xtwxi
    for (int j = 0; j < 3; ++j) {
        const int j3 = j * 3;
        for (int k = 0; k < 3; ++k) {
            xtwxi[j3 + k] = 0.0;
            for (int l = 0; l < 3; ++l)
                xtwxi[j3 + k] += vlr[j * 3 + l] * xtwx[l * 3 + k];
        }
    }

    // compute wy matrix
    for (int j = 0; j < la; ++j) {
        const int jl = j * la;
        wylr[j] = 0.0;
        for (int l = 0; l < la; ++l)
            wylr[j] += wlr[jl + l] * ylr[l];
    }

    // compute xtwy matrix
    for (int j = 0; j < 3; ++j) {
        const int jl = j * la;
        xtwy[j] = 0.0;
        for (int l = 0; l < la; ++l)
            xtwy[j] += xtlr[jl + l] * wylr[l];
    }

    // compute b matrix
    for (int j = 0; j < 3; ++j) {
        const int j3 = j * 3;
        blr[j] = 0.0;
        for (int l = 0; l < 3; ++l)
            blr[j] += xtwxi[j3 + l] * xtwy[l];
    }

    free(xlr);
    free(wlr);
    free(ylr);
    free(wxlr);
    free(xtlr);
    free(wylr);

    return std::min(std::max(int(blr[0] + blr[1] * (radius - cy) + blr[2] * (radius - cx) + 0.5), 0), 255);
}


template <typename PixelType>
static void ProcessFrameD2_Mean(const VSFrameRef *src, const VSFrameRef *pp, VSFrameRef *dst, const TBilateralData *d, const VSAPI *vsapi) {
    const int pixel_max = d->pixel_max;

    for (int j = 0; j < d->vi->format->numPlanes; ++j) {
        if (!d->process[j])
            continue;

        const PixelType *srcp = (const PixelType *)vsapi->getReadPtr(src, j);
        PixelType *dstp = (PixelType *)vsapi->getWritePtr(dst, j);
        const int src_pitch = vsapi->getStride(src, j) / sizeof(PixelType);
        const int dst_pitch = vsapi->getStride(dst, j) / sizeof(PixelType);
        const PixelType *tp = (const PixelType *)vsapi->getReadPtr(pp, j);
        const int tp_pitch = vsapi->getStride(pp, j) / sizeof(PixelType);
        const int width = vsapi->getFrameWidth(src, j);
        const int height = vsapi->getFrameHeight(src, j);
        const int diameter = d->diameter[j];
        const int radius = diameter >> 1;
        int stopy = radius;
        int startyr = radius * diameter;
        const double *spatialWeights = d->spatialWeights[j];
        const double *diffWeights = d->diffWeights[j] + pixel_max * 2;
        const PixelType *srcp_saved = srcp;
        int starty = 0;
        const int midP = width - radius;
        const int midPY = height - radius;
        int y;
        for (y = 0; y < radius; ++y, startyr -= diameter, ++stopy) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }
        for (; y < midPY; ++y, ++starty, ++stopy) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            const PixelType *srcpT2_saved = srcp_saved + stopy * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) // free of all boundaries
            {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const PixelType *srcpT2 = srcpT2_saved;
                const int cP = tp[x] << 1;
                int w = 0;
                for (int u = starty; u <= stopy; ++u) {
                    int b = stopx;
                    for (int v = startx; v <= stopx; ++v, --b, ++w) {
                        const double weight = spatialWeights[w] * diffWeights[cP - srcpT[v] - srcpT2[b]];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                    srcpT2 -= src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight =
                            spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }
        for (--stopy; y < height; ++y, ++starty) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight =
                            spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight =
                            spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight =
                            spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }
    }
}


template <typename PixelType>
static void ProcessFrameD2_MLR(const VSFrameRef *src, const VSFrameRef *pp, VSFrameRef *dst, const TBilateralData *d, const VSAPI *vsapi) {
    const int pixel_max = d->pixel_max;

    for (int j = 0; j < d->vi->format->numPlanes; ++j) {
        if (!d->process[j])
            continue;

        const PixelType *srcp = (const PixelType *)vsapi->getReadPtr(src, j);
        PixelType *dstp = (PixelType *)vsapi->getWritePtr(dst, j);
        const int src_pitch = vsapi->getStride(src, j) / sizeof(PixelType);
        const int dst_pitch = vsapi->getStride(dst, j) / sizeof(PixelType);
        const PixelType *tp = (const PixelType *)vsapi->getReadPtr(pp, j);
        const int tp_pitch = vsapi->getStride(pp, j) / sizeof(PixelType);
        const int width = vsapi->getFrameWidth(src, j);
        const int height = vsapi->getFrameHeight(src, j);
        const int diameter = d->diameter[j];
        const int radius = diameter >> 1;
        int stopy = radius;
        int startyr = radius * diameter;
        const double *spatialWeights = d->spatialWeights[j];
        const double *diffWeights = d->diffWeights[j] + pixel_max * 2;

        const size_t wda = diameter * diameter * sizeof(double);

        double *pixels = vs_aligned_malloc<double>(wda, 16);
        double *weights = vs_aligned_malloc<double>(wda, 16);

        const PixelType *srcp_saved = srcp;
        int starty = 0;
        const int midP = width - radius;
        const int midPY = height - radius;
        int y;
        for (y = 0; y < radius; ++y, startyr -= diameter, ++stopy) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        pixels[w + b] = srcpT[v];
                        weights[w + b] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, stopx - startx + 1,
                                   stopy - starty + 1, startxr, radius + starty - y, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        pixels[w + b] = srcpT[v];
                        weights[w + b] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, diameter,
                                   stopy - starty + 1, 0, radius + starty - y, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        pixels[w + b] = srcpT[v];
                        weights[w + b] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, stopx - startx + 1,
                                   stopy - starty + 1, startxr, radius + starty - y, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }
        for (; y < midPY; ++y, ++starty, ++stopy) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            const PixelType *srcpT2_saved = srcp_saved + stopy * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        pixels[w + b] = srcpT[v];
                        weights[w + b] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, stopx - startx + 1,
                                   diameter, startxr, 0, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) // free of all boundaries
            {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const PixelType *srcpT2 = srcpT2_saved;
                const int cP = tp[x] << 1;
                int w = 0;
                for (int u = starty; u <= stopy; ++u) {
                    int b = stopx;
                    for (int v = startx; v <= stopx; ++v, --b, ++w) {
                        const double weight = spatialWeights[w] * diffWeights[cP - srcpT[v] - srcpT2[b]];
                        pixels[w] = srcpT[v];
                        weights[w] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                    srcpT2 -= src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, diameter,
                                   diameter, 0, 0, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        pixels[w + b] = srcpT[v] * weight;
                        weights[w + b] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, stopx - startx + 1,
                                   diameter, startxr, 0, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }
        for (--stopy; y < height; ++y, ++starty) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        pixels[w + b] = srcpT[v];
                        weights[w + b] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, stopx - startx + 1,
                                   stopy - starty + 1, startxr, radius + starty - y, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        pixels[w + b] = srcpT[v];
                        weights[w + b] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, diameter,
                                   stopy - starty + 1, 0, radius + starty - y, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        pixels[w + b] = srcpT[v];
                        weights[w + b] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, stopx - startx + 1,
                                   stopy - starty + 1, startxr, radius + starty - y, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }

        vs_aligned_free(pixels);
        vs_aligned_free(weights);
    }
}


template <typename PixelType>
static void ProcessFrameD2_Med(const VSFrameRef *src, const VSFrameRef *pp, VSFrameRef *dst, const TBilateralData *d, const VSAPI *vsapi) {
    const bool cw = d->resType == CWMedian;

    const int pixel_max = d->pixel_max;

    size_t medAsize = (pixel_max + 1) * sizeof(double);
    double *medA = vs_aligned_malloc<double>(medAsize, 16);

    for (int j = 0; j < d->vi->format->numPlanes; ++j) {
        if (!d->process[j])
            continue;

        const PixelType *srcp = (const PixelType *)vsapi->getReadPtr(src, j);
        PixelType *dstp = (PixelType *)vsapi->getWritePtr(dst, j);
        const int src_pitch = vsapi->getStride(src, j) / sizeof(PixelType);
        const int dst_pitch = vsapi->getStride(dst, j) / sizeof(PixelType);
        const PixelType *tp = (const PixelType *)vsapi->getReadPtr(pp, j);
        const int tp_pitch = vsapi->getStride(pp, j) / sizeof(PixelType);
        const int width = vsapi->getFrameWidth(src, j);
        const int height = vsapi->getFrameHeight(src, j);
        const int diameter = d->diameter[j];
        const int radius = diameter >> 1;
        int stopy = radius;
        int startyr = radius * diameter;
        const double *spatialWeights = d->spatialWeights[j];
        const double *diffWeights = d->diffWeights[j] + pixel_max * 2;
        const PixelType *srcp_saved = srcp;
        int starty = 0;
        const int midP = width - radius;
        const int midPY = height - radius;
        const int mid = diameter * radius + radius;
        const double cw_weight = spatialWeights[mid] * diffWeights[-pixel_max * 2] * (diameter - 1);
        int y;

        for (y = 0; y < radius; ++y, startyr -= diameter, ++stopy) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }
        for (; y < midPY; ++y, ++starty, ++stopy) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            const PixelType *srcpT2_saved = srcp_saved + stopy * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) // free of all boundaries
            {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const PixelType *srcpT2 = srcpT2_saved;
                const int cP = tp[x] << 1;
                int w = 0;
                for (int u = starty; u <= stopy; ++u) {
                    int b = stopx;
                    for (int v = startx; v <= stopx; ++v, --b, ++w) {
                        const double weight = spatialWeights[w] * diffWeights[cP - srcpT[v] - srcpT2[b]];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                    srcpT2 -= src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }
        for (--stopy; y < height; ++y, ++starty) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[(cP - srcpT[v]) << 1];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }
    }

    vs_aligned_free(medA);
}


template <typename PixelType>
static void ProcessFrameD1_Mean(const VSFrameRef *src, const VSFrameRef *pp, VSFrameRef *dst, const TBilateralData *d, const VSAPI *vsapi) {
    const int pixel_max = d->pixel_max;

    for (int j = 0; j < d->vi->format->numPlanes; ++j) {
        if (!d->process[j])
            continue;

        const PixelType *srcp = (const PixelType *)vsapi->getReadPtr(src, j);
        PixelType *dstp = (PixelType *)vsapi->getWritePtr(dst, j);
        const int src_pitch = vsapi->getStride(src, j) / sizeof(PixelType);
        const int dst_pitch = vsapi->getStride(dst, j) / sizeof(PixelType);
        const PixelType *tp = (const PixelType *)vsapi->getReadPtr(pp, j);
        const int tp_pitch = vsapi->getStride(pp, j) / sizeof(PixelType);
        const int width = vsapi->getFrameWidth(src, j);
        const int height = vsapi->getFrameHeight(src, j);
        const int diameter = d->diameter[j];
        const int radius = diameter >> 1;
        int stopy = radius;
        int startyr = radius * diameter;
        const double *spatialWeights = d->spatialWeights[j];
        const double *diffWeights = d->diffWeights[j] + pixel_max;
        const PixelType *srcp_saved = srcp;
        int starty = 0;
        const int midP = width - radius;
        const int midPY = height - radius;
        int y;
        for (y = 0; y < radius; ++y, startyr -= diameter, ++stopy) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }
        for (; y < midPY; ++y, ++starty, ++stopy) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) // free of all boundaries
            {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = 0;
                for (int u = starty; u <= stopy; ++u) {
                    for (int v = startx; v <= stopx; ++v, ++w) {
                        const double weight = spatialWeights[w] * diffWeights[cP - srcpT[v]];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }
        for (--stopy; y < height; ++y, ++starty) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double weightedSum = 0.0;
                double sumOfWeights = 0.0;
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        weightedSum += srcpT[v] * weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = int((weightedSum / sumOfWeights) + 0.5);
                else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }
    }
}


template <typename PixelType>
static void ProcessFrameD1_MLR(const VSFrameRef *src, const VSFrameRef *pp, VSFrameRef *dst, const TBilateralData *d, const VSAPI *vsapi) {
    const int pixel_max = d->pixel_max;

    for (int j = 0; j < d->vi->format->numPlanes; ++j) {
        if (!d->process[j])
            continue;

        const PixelType *srcp = (const PixelType *)vsapi->getReadPtr(src, j);
        PixelType *dstp = (PixelType *)vsapi->getWritePtr(dst, j);
        const int src_pitch = vsapi->getStride(src, j) / sizeof(PixelType);
        const int dst_pitch = vsapi->getStride(dst, j) / sizeof(PixelType);
        const PixelType *tp = (const PixelType *)vsapi->getReadPtr(pp, j);
        const int tp_pitch = vsapi->getStride(pp, j) / sizeof(PixelType);
        const int width = vsapi->getFrameWidth(src, j);
        const int height = vsapi->getFrameHeight(src, j);
        const int diameter = d->diameter[j];
        const int radius = diameter >> 1;
        int stopy = radius;
        int startyr = radius * diameter;
        const double *spatialWeights = d->spatialWeights[j];
        const double *diffWeights = d->diffWeights[j] + pixel_max;

        const size_t wda = diameter * diameter * sizeof(double);

        double *pixels = vs_aligned_malloc<double>(wda, 16);
        double *weights = vs_aligned_malloc<double>(wda, 16);

        const PixelType *srcp_saved = srcp;
        int starty = 0;
        const int midP = width - radius;
        const int midPY = height - radius;
        int y;
        for (y = 0; y < radius; ++y, startyr -= diameter, ++stopy) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        pixels[w + b] = srcpT[v];
                        weights[w + b] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, stopx - startx + 1,
                                   stopy - starty + 1, startxr, radius + starty - y, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        pixels[w + b] = srcpT[v];
                        weights[w + b] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, diameter,
                                   stopy - starty + 1, 0, radius + starty - y, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        pixels[w + b] = srcpT[v];
                        weights[w + b] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, stopx - startx + 1,
                                   stopy - starty + 1, startxr, radius + starty - y, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }
        for (; y < midPY; ++y, ++starty, ++stopy) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        pixels[w + b] = srcpT[v];
                        weights[w + b] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, stopx - startx + 1,
                                   diameter, startxr, 0, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) // free of all boundaries
            {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = 0;
                for (int u = starty; u <= stopy; ++u) {
                    for (int v = startx; v <= stopx; ++v, ++w) {
                        const double weight = spatialWeights[w] * diffWeights[cP - srcpT[v]];
                        pixels[w] = srcpT[v];
                        weights[w] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, diameter,
                                   diameter, 0, 0, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        pixels[w + b] = srcpT[v];
                        weights[w + b] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, stopx - startx + 1,
                                   diameter, startxr, 0, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }
        for (--stopy; y < height; ++y, ++starty) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        pixels[w + b] = srcpT[v];
                        weights[w + b] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, stopx - startx + 1,
                                   stopy - starty + 1, startxr, radius + starty - y, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        pixels[w + b] = srcpT[v];
                        weights[w + b] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, diameter,
                                   stopy - starty + 1, 0, radius + starty - y, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double sumOfWeights = 0.0;
                memset(pixels, 0, wda);
                memset(weights, 0, wda);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        pixels[w + b] = srcpT[v];
                        weights[w + b] = weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS)
                    dstp[x] = mlre(pixels, weights, stopx - startx + 1,
                                   stopy - starty + 1, startxr, radius + starty - y, radius, diameter);
                else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }

        vs_aligned_free(pixels);
        vs_aligned_free(weights);
    }
}


template <typename PixelType>
static void ProcessFrameD1_Med(const VSFrameRef *src, const VSFrameRef *pp, VSFrameRef *dst, const TBilateralData *d, const VSAPI *vsapi) {
    const bool cw = d->resType == CWMedian;

    const int pixel_max = d->pixel_max;

    size_t medAsize = (pixel_max + 1) * sizeof(double);
    double *medA = vs_aligned_malloc<double>(medAsize, 16);

    for (int j = 0; j < d->vi->format->numPlanes; ++j) {
        if (!d->process[j])
            continue;

        const PixelType *srcp = (const PixelType *)vsapi->getReadPtr(src, j);
        PixelType *dstp = (PixelType *)vsapi->getWritePtr(dst, j);
        const int src_pitch = vsapi->getStride(src, j) / sizeof(PixelType);
        const int dst_pitch = vsapi->getStride(dst, j) / sizeof(PixelType);
        const PixelType *tp = (const PixelType *)vsapi->getReadPtr(pp, j);
        const int tp_pitch = vsapi->getStride(pp, j) / sizeof(PixelType);
        const int width = vsapi->getFrameWidth(src, j);
        const int height = vsapi->getFrameHeight(src, j);
        const int diameter = d->diameter[j];
        const int radius = diameter >> 1;
        int stopy = radius;
        int startyr = radius * diameter;
        const double *spatialWeights = d->spatialWeights[j];
        const double *diffWeights = d->diffWeights[j] + pixel_max;
        const PixelType *srcp_saved = srcp;
        int starty = 0;
        const int midP = width - radius;
        const int midPY = height - radius;
        const int mid = diameter * radius + radius;
        const double cw_weight = spatialWeights[mid] * diffWeights[-pixel_max] * (diameter - 1);
        int y;
        for (y = 0; y < radius; ++y, startyr -= diameter, ++stopy) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }
        for (; y < midPY; ++y, ++starty, ++stopy) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) // free of all boundaries
            {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = 0;
                for (int u = starty; u <= stopy; ++u) {
                    for (int v = startx; v <= stopx; ++v, ++w) {
                        const double weight = spatialWeights[w] * diffWeights[cP - srcpT[v]];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }
        for (--stopy; y < height; ++y, ++starty) {
            const PixelType *srcpT_saved = srcp_saved + starty * src_pitch;
            int startx = 0;
            int startxr = radius;
            int stopx = radius;
            int x;
            for (x = 0; x < radius; ++x, --startxr, ++stopx) {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            for (; x < midP; ++x, ++startx, ++stopx) {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            for (--stopx; x < width; ++x, ++startx) {
                double sumOfWeights = 0.0;
                double sum = 0.0;
                memset(medA, 0, medAsize);
                const PixelType *srcpT = srcpT_saved;
                const int cP = tp[x];
                int w = startyr;
                for (int u = starty; u <= stopy; ++u, w += diameter) {
                    int b = startxr;
                    for (int v = startx; v <= stopx; ++v, ++b) {
                        const double weight = spatialWeights[w + b] * diffWeights[cP - srcpT[v]];
                        medA[srcpT[v]] += weight;
                        sumOfWeights += weight;
                    }
                    srcpT += src_pitch;
                }
                if (sumOfWeights >= MINS) {
                    if (cw) {
                        medA[tp[x]] += cw_weight;
                        sumOfWeights += cw_weight;
                    }
                    sumOfWeights *= 0.5;
                    int ws = 0;
                    while (sum <= sumOfWeights) {
                        sum += medA[ws];
                        ++ws;
                    }
                    dstp[x] = ws - 1;
                } else
                    dstp[x] = srcp[x];
            }
            srcp += src_pitch;
            dstp += dst_pitch;
            tp += tp_pitch;
        }
    }

    vs_aligned_free(medA);
}


static void VS_CC TBilateralInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    (void)in;
    (void)out;
    (void)core;

    TBilateralData *d = (TBilateralData *) *instanceData;

    vsapi->setVideoInfo(d->vi, 1, node);
}


static const VSFrameRef *VS_CC TBilateralGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    (void)frameData;

    const TBilateralData *d = (const TBilateralData *) *instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->clip, frameCtx);
        vsapi->requestFrameFilter(n, d->ppclip, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(n, d->clip, frameCtx);
        const VSFrameRef *pp = vsapi->getFrameFilter(n, d->ppclip, frameCtx);

        const VSFrameRef *plane_src[3] = {
            d->process[0] ? nullptr : src,
            d->process[1] ? nullptr : src,
            d->process[2] ? nullptr : src
        };

        int planes[3] = { 0, 1, 2 };

        VSFrameRef *dst = vsapi->newVideoFrame2(d->vi->format, d->vi->width, d->vi->height, plane_src, planes, src, core);

        d->process_frame(src, pp, dst, d, vsapi);

        vsapi->freeFrame(src);
        vsapi->freeFrame(pp);

        return dst;
    }

    return nullptr;
}


static void VS_CC TBilateralFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    (void)core;

    TBilateralData *d = (TBilateralData *)instanceData;

    freeTables(d);

    vsapi->freeNode(d->clip);
    vsapi->freeNode(d->ppclip);
    free(d);
}


#define TBILATERAL_FILTER "TBilateral"

static void VS_CC TBilateralCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    (void)userData;

    TBilateralData d;
    memset(&d, 0, sizeof(d));

    int err;

    for (int i = 0; i < 3; i++) {
        d.diameter[i] = int64ToIntS(vsapi->propGetInt(in, "diameter", i, &err));
        if (err)
            d.diameter[i] = (i == 0) ? 5
                                     : d.diameter[i - 1];

        if (d.diameter[i] <= 1 || d.diameter[i] % 2 == 0) {
            vsapi->setError(out, TBILATERAL_FILTER ": diameter must be an odd number greater than 1.");
            return;
        }


        d.sDev[i] = vsapi->propGetFloat(in, "sdev", i, &err);
        if (err)
            d.sDev[i] = (i == 0) ? 1.4
                                 : d.sDev[i - 1];

        if (d.sDev[i] < 0) {
            vsapi->setError(out, TBILATERAL_FILTER ": sdev must be at least 0.");
            return;
        }


        d.iDev[i] = vsapi->propGetFloat(in, "idev", i, &err);
        if (err)
            d.iDev[i] = (i == 0) ? 7.0
                                 : d.iDev[i - 1];

        if (d.iDev[i] < 0) {
            vsapi->setError(out, TBILATERAL_FILTER ": idev must be at least 0.");
            return;
        }


        d.cs[i] = vsapi->propGetFloat(in, "cs", i, &err);
        if (err)
            d.cs[i] = (i == 0) ? 1.0
                               : d.cs[i - 1];

        if (d.cs[i] < 0) {
            vsapi->setError(out, TBILATERAL_FILTER ": cs must be at least 0.");
            return;
        }
    }


    d.d2 = !!vsapi->propGetInt(in, "d2", 0, &err);
    if (err)
        d.d2 = false;


    d.kernS = int64ToIntS(vsapi->propGetInt(in, "kerns", 0, &err));
    if (err)
        d.kernS = Gaussian;

    if (d.kernS < AndrewsWave || d.kernS > Inverse) {
        vsapi->setError(out, TBILATERAL_FILTER ": kerns must be between 0 and 9 (inclusive).");
        return;
    }


    d.kernI = int64ToIntS(vsapi->propGetInt(in, "kerni", 0, &err));
    if (err)
        d.kernI = Gaussian;

    if (d.kernI < AndrewsWave || d.kernI > Inverse) {
        vsapi->setError(out, TBILATERAL_FILTER ": kerni must be between 0 and 9 (inclusive).");
        return;
    }


    d.resType = int64ToIntS(vsapi->propGetInt(in, "restype", 0, &err));
    if (err)
        d.resType = Mean;

    if (d.resType < Mean || d.resType > MultipleLinearRegression) {
        vsapi->setError(out, TBILATERAL_FILTER ": restype must be between 0 and 3 (inclusive).");
        return;
    }


    d.clip = vsapi->propGetNode(in, "clip", 0, nullptr);
    d.vi = vsapi->getVideoInfo(d.clip);

    if (!d.vi->format) {
        vsapi->setError(out, TBILATERAL_FILTER ": clip must have constant format.");
        vsapi->freeNode(d.clip);
        return;
    }

    if (d.vi->format->bitsPerSample > 16 || d.vi->format->sampleType != stInteger) {
        vsapi->setError(out, TBILATERAL_FILTER ": clip must be 8..16 bit with integer sample type.");
        vsapi->freeNode(d.clip);
        return;
    }

    if (d.vi->width == 0 || d.vi->height == 0) {
        vsapi->setError(out, TBILATERAL_FILTER ": clip must have constant dimensions.");
        vsapi->freeNode(d.clip);
        return;
    }

    if (d.vi->width % 2 == 1 || d.vi->height % 2 == 1) {
        vsapi->setError(out, TBILATERAL_FILTER ": clip's dimensions must be multiples of 2.");
        vsapi->freeNode(d.clip);
        return;
    }

    int width[3] = {
        d.vi->width,
        d.vi->width >> d.vi->format->subSamplingW,
        d.vi->width >> d.vi->format->subSamplingW
    };
    int height[3] = {
        d.vi->height,
        d.vi->height >> d.vi->format->subSamplingH,
        d.vi->height >> d.vi->format->subSamplingH
    };

    for (int i = 0; i < d.vi->format->numPlanes; i++) {
        if (d.diameter[i] > width[i] || d.diameter[i] > height[i]) {
#define ERROR_SIZE 71
            char error[ERROR_SIZE] = { 0 };
            snprintf(error, ERROR_SIZE, TBILATERAL_FILTER ": diameter[%d] must be less than the dimensions of plane %d.", i, i);
#undef ERROR_SIZE

            vsapi->setError(out, error);
            vsapi->freeNode(d.clip);
            return;
        }
    }

    int n = d.vi->format->numPlanes;
    int m = vsapi->propNumElements(in, "planes");

    for (int i = 0; i < 3; i++)
        d.process[i] = (m <= 0);

    for (int i = 0; i < m; i++) {
        int o = int64ToIntS(vsapi->propGetInt(in, "planes", i, nullptr));

        if (o < 0 || o >= n) {
            vsapi->freeNode(d.clip);
            vsapi->setError(out, TBILATERAL_FILTER ": plane index out of range");
            return;
        }

        if (d.process[o]) {
            vsapi->freeNode(d.clip);
            vsapi->setError(out, TBILATERAL_FILTER ": plane specified twice");
            return;
        }

        d.process[o] = 1;
    }


    d.ppclip = vsapi->propGetNode(in, "ppclip", 0, &err);
    if (err)
        d.ppclip = vsapi->cloneNodeRef(d.clip);

    const VSVideoInfo *ppvi = vsapi->getVideoInfo(d.ppclip);

    if (d.vi->format != ppvi->format) {
        vsapi->setError(out, TBILATERAL_FILTER ": clip and ppclip must have the same format.");
        vsapi->freeNode(d.clip);
        vsapi->freeNode(d.ppclip);
        return;
    }

    if (d.vi->width != ppvi->width || d.vi->height != ppvi->height) {
        vsapi->setError(out, TBILATERAL_FILTER ": clip and ppclip must have the same dimensions.");
        vsapi->freeNode(d.clip);
        vsapi->freeNode(d.ppclip);
        return;
    }

    if (d.vi->numFrames != ppvi->numFrames) {
        vsapi->setError(out, TBILATERAL_FILTER ": clip and ppclip must have the same number of frames.");
        vsapi->freeNode(d.clip);
        vsapi->freeNode(d.ppclip);
        return;
    }


    int bits = d.vi->format->bitsPerSample;

    if (d.d2) {
        if (d.resType == Mean)
            d.process_frame = bits == 8 ? ProcessFrameD2_Mean<uint8_t>
                                        : ProcessFrameD2_Mean<uint16_t>;
        else if (d.resType == MultipleLinearRegression)
            d.process_frame = bits == 8 ? ProcessFrameD2_MLR<uint8_t>
                                        : ProcessFrameD2_MLR<uint16_t>;
        else
            d.process_frame = bits == 8 ? ProcessFrameD2_Med<uint8_t>
                                        : ProcessFrameD2_Med<uint16_t>;
    } else {
        if (d.resType == Mean)
            d.process_frame = bits == 8 ? ProcessFrameD1_Mean<uint8_t>
                                        : ProcessFrameD1_Mean<uint16_t>;
        else if (d.resType == MultipleLinearRegression)
            d.process_frame = bits == 8 ? ProcessFrameD1_MLR<uint8_t>
                                        : ProcessFrameD1_MLR<uint16_t>;
        else
            d.process_frame = bits == 8 ? ProcessFrameD1_Med<uint8_t>
                                        : ProcessFrameD1_Med<uint16_t>;
    }


    d.pixel_max = (1 << bits) - 1;

    for (int i = 0; i < 3; i++)
        d.iDev[i] = d.iDev[i] * d.pixel_max / 255;


    buildTables(&d);


    TBilateralData *data = (TBilateralData *)malloc(sizeof(d));
    *data = d;

    vsapi->createFilter(in, out, TBILATERAL_FILTER, TBilateralInit, TBilateralGetFrame, TBilateralFree, fmParallel, 0, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.nodame.tbilateral", "tbilateral", "Bilateral spatial denoising filter", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc(TBILATERAL_FILTER,
                 "clip:clip;"
                 "ppclip:clip:opt;"
                 "diameter:int[]:opt;"
                 "sdev:float[]:opt;"
                 "idev:float[]:opt;"
                 "cs:float[]:opt;"
                 "d2:int:opt;"
                 "kerns:int:opt;"
                 "kerni:int:opt;"
                 "restype:int:opt;"
                 "planes:int[]:opt;"
                 , TBilateralCreate, nullptr, plugin);
}
