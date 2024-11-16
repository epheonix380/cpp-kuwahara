//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//    Code examples that show how class MultiPartInputFile and
//    class MultiPartOutputFile can be used to read and write
//    OpenEXR image files containing multiple images.
//
//-----------------------------------------------------------------------------

#include <ImfHeader.h>
#include <ImfPartType.h>
#include <ImfChannelList.h>
#include <ImfArray.h>

#include <ImfFrameBuffer.h>
#include <ImfDeepFrameBuffer.h>

#include <ImfMultiPartInputFile.h>
#include <ImfMultiPartOutputFile.h>

#include <ImfInputPart.h>
#include <ImfTiledInputPart.h>
#include <ImfDeepScanLineInputPart.h>
#include <ImfDeepTiledInputPart.h>

#include <ImfOutputPart.h>
#include <ImfTiledOutputPart.h>
#include <ImfDeepScanLineOutputPart.h>
#include <ImfDeepTiledOutputPart.h>

#include <vector>
#include <list>

#include "namespaceAlias.h"
#include <WindowsNumerics.h>
#include <cstring>
#include <thread>
#include <iostream>
using namespace Windows::Foundation::Numerics;
using namespace IMF;
using namespace std;
using namespace IMATH_NAMESPACE;
#define PI 3.14159265358979323846

template <class InputPartType, class OutputPartType>
void
copyPixels (
    MultiPartInputFile&  inputFile,
    MultiPartOutputFile& outputFile,
    int                  inputPartNumber,
    int                  outputPartNumber)
{
    InputPartType  inPart (inputFile, inputPartNumber);
    OutputPartType outPart (outputFile, outputPartNumber);
    outPart.copyPixels (inPart);
}

void
copyPixels (
    const std::string&   type,
    MultiPartInputFile&  inputFile,
    MultiPartOutputFile& outputFile,
    int                  inputPartNumber,
    int                  outputPartNumber)
{
    //
    // Copy pixels from a given part of a multipart input file
    // to a given part of a multipart output file.
    //

    if (type == SCANLINEIMAGE)
    {
        copyPixels<InputPart, OutputPart> (
            inputFile, outputFile, inputPartNumber, outputPartNumber);
    }
    else if (type == TILEDIMAGE)
    {
        copyPixels<TiledInputPart, TiledOutputPart> (
            inputFile, outputFile, inputPartNumber, outputPartNumber);
    }
    else if (type == DEEPSCANLINE)
    {
        copyPixels<DeepScanLineInputPart, DeepScanLineOutputPart> (
            inputFile, outputFile, inputPartNumber, outputPartNumber);
    }
    else if (type == DEEPTILE)
    {
        copyPixels<DeepTiledInputPart, DeepTiledOutputPart> (
            inputFile, outputFile, inputPartNumber, outputPartNumber);
    }
}

void
combineFiles ()
{
    //
    // Read multiple single-part input files and write them as a multi-part file.
    // If an input file is multi-part, only one part is copied.
    // All input files dimentions must be the same.
    //

    std::vector<MultiPartInputFile*> inputFiles;
    std::vector<Header>              headers;

    const char* filenames[] = {
        "gz1.exr", "tiledgz1.exr", "test.deep.exr", "testTiled.deep.exr"};

    for (size_t i = 0; i < sizeof (filenames) / sizeof (filenames[0]); i++)
    {
        MultiPartInputFile* in_file = new MultiPartInputFile (filenames[i]);
        Header              header  = in_file->header (0);

        if (!header.hasName ()) { header.setName (filenames[i]); }

        inputFiles.push_back (in_file);
        headers.push_back (header);
    }

    MultiPartOutputFile outputFile (
        "multipart.exr", headers.data (), (int) headers.size ());

    for (size_t i = 0; i < sizeof (filenames) / sizeof (filenames[0]); i++)
    {
        Header&       header = headers[i];
        const string& type   = header.type ();
        copyPixels (type, *inputFiles[i], outputFile, 0, i);
        delete inputFiles[i];
    }
}

void
splitFiles ()
{
    //
    // Read a multi-part input file and write all parts as separate files.
    //

    MultiPartInputFile inputFile ("modified.exr");

    for (int i = 0; i < inputFile.parts (); i++)
    {
        Header header = inputFile.header (i);
        string out_path =
            string ("split_part_") + to_string (i) + string (".exr");
        const string&       type = header.type ();
        MultiPartOutputFile outputFile (out_path.c_str (), &header, 1);
        copyPixels (type, inputFile, outputFile, i, 0);
    }
}

template <typename ValueType>
void
insertSlice (
    Box2i&                    dataWindow,
    FrameBuffer&              frameBuffer,
    const char                name[],
    PixelType                 pixelType,
    list<Array2D<ValueType>>& channels)
{
    //
    // Allocate a pixel buffer and describe the layout of
    // the buffer in the FrameBuffer object.
    //

    size_t width  = dataWindow.max.x - dataWindow.min.x + 1;
    size_t height = dataWindow.max.y - dataWindow.min.y + 1;

    channels.emplace_back ();
    Array2D<ValueType>& buffer = channels.back ();
    buffer.resizeErase (height, width);

    char* base =
        (char*) (&buffer[0][0] - dataWindow.min.x - dataWindow.min.y * width);

    frameBuffer.insert (
        name, // name
        Slice (
            pixelType,                 // type
            base,                      // base
            sizeof (ValueType) * 1,    // xStride
            sizeof (ValueType) * width // yStride
            ));
}

template <typename ValueType>
void
insertDeepSlice (
    Box2i&                     dataWindow,
    DeepFrameBuffer&           frameBuffer,
    const char                 name[],
    PixelType                  pixelType,
    list<Array2D<ValueType*>>& channels)
{
    //
    // Allocate a pixel buffer and describe the layout of
    // the buffer in the DeepFrameBuffer object.
    //

    size_t width  = dataWindow.max.x - dataWindow.min.x + 1;
    size_t height = dataWindow.max.y - dataWindow.min.y + 1;

    channels.emplace_back ();
    Array2D<ValueType*>& buffer = channels.back ();
    buffer.resizeErase (height, width);

    char* base =
        (char*) (&buffer[0][0] - dataWindow.min.x - dataWindow.min.y * width);

    frameBuffer.insert (
        name, // name
        DeepSlice (
            pixelType,                   // type
            base,                        // base
            sizeof (ValueType*) * 1,     // xStride
            sizeof (ValueType*) * width, // yStride
            sizeof (ValueType)           // sample stride
            ));
}

FrameBuffer
setupFramebuffer (
    const Header&            header,
    list<Array2D<uint32_t>>& intChannels,
    list<Array2D<half>>&     halfChannels,
    list<Array2D<float>>&    floatChannels)
{
    //
    // Allocate pixel buffers for all channels specified in the header, describe the layout of
    // the buffers in the FrameBuffer object.
    //

    FrameBuffer frameBuffer;
    Box2i       dataWindow = header.dataWindow ();

    for (auto i = header.channels ().begin (); i != header.channels ().end ();
         i++)
    {
        if (i.channel ().type == UINT)
        {
            insertSlice<uint32_t> (
                dataWindow,
                frameBuffer,
                i.name (),
                i.channel ().type,
                intChannels);
        }
        else if (i.channel ().type == HALF)
        {
            insertSlice<half> (
                dataWindow,
                frameBuffer,
                i.name (),
                i.channel ().type,
                halfChannels);
        }
        else if (i.channel ().type == FLOAT)
        {
            insertSlice<float> (
                dataWindow,
                frameBuffer,
                i.name (),
                i.channel ().type,
                floatChannels);
        }
    }

    return frameBuffer;
}

DeepFrameBuffer
setupDeepFramebuffer (
    const Header&             header,
    Array2D<uint32_t>&        sampleCount,
    list<Array2D<uint32_t*>>& intChannels,
    list<Array2D<half*>>&     halfChannels,
    list<Array2D<float*>>&    floatChannels)
{
    //
    // Allocate pixel buffers for all channels specified in the header, describe the layout of
    // the buffers in the DeepFrameBuffer object.
    //

    DeepFrameBuffer frameBuffer;

    Box2i dataWindow = header.dataWindow ();

    size_t width  = dataWindow.max.x - dataWindow.min.x + 1;
    size_t height = dataWindow.max.y - dataWindow.min.y + 1;

    sampleCount.resizeErase (height, width);

    frameBuffer.insertSampleCountSlice (Slice (
        UINT,
        (char*) (&sampleCount[0][0] - dataWindow.min.x -
                 dataWindow.min.y * width),
        sizeof (unsigned int) * 1,    // xStride
        sizeof (unsigned int) * width // yStride
        ));

    for (auto i = header.channels ().begin (); i != header.channels ().end ();
         i++)
    {
        if (i.channel ().type == UINT)
        {
            insertDeepSlice<uint32_t> (
                dataWindow,
                frameBuffer,
                i.name (),
                i.channel ().type,
                intChannels);
        }
        else if (i.channel ().type == HALF)
        {
            insertDeepSlice<half> (
                dataWindow,
                frameBuffer,
                i.name (),
                i.channel ().type,
                halfChannels);
        }
        else if (i.channel ().type == FLOAT)
        {
            insertDeepSlice<float> (
                dataWindow,
                frameBuffer,
                i.name (),
                i.channel ().type,
                floatChannels);
        }
    }

    return frameBuffer;
}

template <typename T>
void
resizeDeepBuffers (Array2D<uint32_t>& sampleCount, list<Array2D<T*>>& channels)
{
    //
    // Allocate memory for samples in all pixel buffers according to the data in sampleCount buffer.
    //

    for (auto i = channels.begin (); i != channels.end (); i++)
    {
        Array2D<T*>& channel = *i;
        for (int y = 0; y < channel.height (); y++)
        {
            for (int x = 0; x < channel.width (); x++)
            {
                uint32_t count = sampleCount[y][x];
                if (count)
                    channel[y][x]  = new T[count];
                else
                    channel[y][x]  = nullptr;
            }
        }
    }
}

template <typename T>
void
freeDeepBuffers (list<Array2D<T*>>& channels)
{
    for (auto i = channels.begin (); i != channels.end (); i++)
    {
        Array2D<T*>& channel = *i;
        for (int y = 0; y < channel.height (); y++)
            for (int x = 0; x < channel.width (); x++)
                delete[] channel[y][x];
    }
}

unsigned int
poopy (unsigned short y)
{

    int s = (y >> 15) & 0x00000001;
    int e = (y >> 10) & 0x0000001f;
    int m = y & 0x000003ff;

    if (e == 0)
    {
        if (m == 0)
        {
            //
            // Plus or minus zero
            //

            return s << 31;
        }
        else
        {
            //
            // Denormalized number -- renormalize it
            //

            while (!(m & 0x00000400))
            {
                m <<= 1;
                e -= 1;
            }

            e += 1;
            m &= ~0x00000400;
        }
    }
    else if (e == 31)
    {
        if (m == 0)
        {
            //
            // Positive or negative infinity
            //

            return (s << 31) | 0x7f800000;
        }
        else
        {
            //
            // Nan -- preserve sign and significand bits
            //

            return (s << 31) | 0x7f800000 | (m << 13);
        }
    }

    //
    // Normalized number
    //

    e = e + (127 - 15);
    m = m << 13;

    //
    // Assemble s, e and m.
    //

    return (s << 31) | (e << 23) | m;
}
inline float
square (float x)
{
    return x * x;
}

template <typename T>
void
getTensor (
    Array2D<T>& red,
    Array2D<T>& green,
    Array2D<T>& blue,
    Array2D<T>& alpha,
    Array2D<float>& destR,
    Array2D<float>& destG,
    Array2D<float>& destB
)
{
    const float corner_weight = 0.182f;
    const float center_weight = 1.0f - 2.0f * corner_weight;
        int height = min (
            min (red.height (), blue.height ()),
            min (green.height (), alpha.height ()));
        int width = min (
            min (red.width (), blue.width ()),
            min (green.width (), alpha.width ()));
        cout << width << endl;
        cout << height << endl;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int lowX = max (0, x - 1);
                int hiX  = min (width - 1, x + 1);
                int lowY = max (0, y - 1);
                int hiY  = min (height - 1, y + 1);


                float red1 = (red[hiY][lowX]);
                float red2 = (red[y][lowX]);
                float red3 = (red[lowY][lowX]);
                float red4 = (red[hiY][hiX]);
                float red5 = (red[y][hiX]);
                float red6 = (red[lowY][hiX]);
                float r    = red1 * -corner_weight + red2 * -center_weight +
                          red3 * -corner_weight + red4 * corner_weight +
                          red5 * center_weight + red6 * corner_weight;

                float green1 = (green[hiY][lowX]);
                float green2 = (green[y][lowX]);
                float green3 = (green[lowY][lowX]);
                float green4 = (green[hiY][hiX]);
                float green5 = (green[y][hiX]);
                float green6 = (green[lowY][hiX]);
                float g = green1 * -corner_weight + green2 * -center_weight +
                          green3 * -corner_weight + green4 * corner_weight +
                          green5 * center_weight + green6 * corner_weight;

                float blue1 = (blue[hiY][lowX]);
                float blue2 = (blue[y][lowX]);
                float blue3 = (blue[lowY][lowX]);
                float blue4 = (blue[hiY][hiX]);
                float blue5 = (blue[y][hiX]);
                float blue6 = (blue[lowY][hiX]);
                float b     = blue1 * -corner_weight + blue2 * -center_weight +
                          blue3 * -corner_weight + blue4 * corner_weight +
                          blue5 * center_weight + blue6 * corner_weight;

                float alpha1 = (alpha[hiY][lowX]);
                float alpha2 = (alpha[y][lowX]);
                float alpha3 = (alpha[lowY][lowX]);
                float alpha4 = (alpha[hiY][hiX]);
                float alpha5 = (alpha[y][hiX]);
                float alpha6 = (alpha[lowY][hiX]);
                float a = alpha1 * -corner_weight + alpha2 * -center_weight +
                          alpha3 * -corner_weight + alpha4 * corner_weight +
                          alpha5 * center_weight + alpha6 * corner_weight;

                float4 x_partial_derivative = float4 (r, g, b, a);

                float yR = red[hiY][lowX] * corner_weight +
                           red[hiY][x] * center_weight +
                           red[hiY][hiX] * corner_weight +
                           red[lowY][lowX] * -corner_weight +
                           red[lowY][x] * -center_weight +
                           red[lowY][hiX] * -corner_weight;

                float yG = green[hiY][lowX] * corner_weight +
                           green[hiY][x] * center_weight +
                           green[hiY][hiX] * corner_weight +
                           green[lowY][lowX] * -corner_weight +
                           green[lowY][x] * -center_weight +
                           green[lowY][hiX] * -corner_weight;

                float yB = blue[hiY][lowX] * corner_weight +
                           blue[hiY][x] * center_weight +
                           blue[hiY][hiX] * corner_weight +
                           blue[lowY][lowX] * -corner_weight +
                           blue[lowY][x] * -center_weight +
                           blue[lowY][hiX] * -corner_weight;

                float yA = alpha[hiY][lowX] * corner_weight +
                           alpha[hiY][x] * center_weight +
                           alpha[hiY][hiX] * corner_weight +
                           alpha[lowY][lowX] * -corner_weight +
                           alpha[lowY][x] * -center_weight +
                           alpha[lowY][hiX] * -corner_weight;

                float4 y_partial_derivative = float4 (yR, yG, yB, yA);

                float dxdx = dot (x_partial_derivative, x_partial_derivative);
                float dxdy = dot (x_partial_derivative, y_partial_derivative);
                float dydy = dot (y_partial_derivative, y_partial_derivative);
                destR[y][x] = (T) dxdx;
                destG[y][x] = (T) dxdy;
                destB[y][x] = (T) dydy;
            }
        }
    
}

template <typename T>
void
getMap (
    Array2D<float>& red,
    Array2D<float>& green,
    Array2D<float>& blue,
    Array2D<T>& alphaMap,
    Array2D<float>& destR,
    Array2D<float>& destG,
    Array2D<float>& destB,
    Array2D<float>& destA
    )
{
        const float corner_weight = 0.182f;
        const float center_weight = 1.0f - 2.0f * corner_weight;
        int         height        = min (
            min (red.height (), blue.height ()), green.height ());
        int width = min (
            min (red.width (), blue.width ()), green.width ());
        cout << width << endl;
        cout << height << endl;

        for (int y = 0; y < height; y++)
        {
            cout << y << endl;
            for (int x = 0; x < width; x++)
            {

                int   lowX = max (0, x - 1);
                int   hiX  = min (width - 1, x + 1);
                int   lowY = max (0, y - 1);
                int   hiY  = min (height - 1, y + 1);
                float dxdx = red[y][x];
                float dxdy = blue[y][x];
                float dydy = green[y][x];

                // Compute the first and second eigenvalues of the structure tensor using the equations in
                // section "3.1 Orientation and Anisotropy Estimation" of the paper.
                float eigenvalue_first_term = (dxdx + dydy) / 2.f;
                float eigenvalue_square_root_term =
                    sqrt (square (dxdx - dydy) + 4.0f * square (dxdy)) / 2.0f;
                float first_eigenvalue =
                    eigenvalue_first_term + eigenvalue_square_root_term;
                float second_eigenvalue =
                    eigenvalue_first_term - eigenvalue_square_root_term;

                // Compute the normalized eigenvector of the structure tensor oriented in direction of the
                // minimum rate of change using the equations in section "3.1 Orientation and Anisotropy
                // Estimation" of the paper.
                float2 eigenvector =
                    float2 (first_eigenvalue - dxdx, -1.0f * dxdy);
                float  eigenvector_length = length (eigenvector);
                float2 unit_eigenvector   = eigenvector_length != 0.0f
                                                ? eigenvector / eigenvector_length
                                                : float2 (1.0f);

                // Compute the amount of anisotropy using equations in section "3.1 Orientation and Anisotropy
                // Estimation" of the paper. The anisotropy ranges from 0 to 1, where 0 corresponds to isotropic
                // and 1 corresponds to entirely anisotropic regions.
                float eigenvalue_sum = first_eigenvalue + second_eigenvalue;
                float eigenvalue_difference =
                    first_eigenvalue - second_eigenvalue;
                float anisotropy = eigenvalue_sum > 0.0f
                                       ? eigenvalue_difference / eigenvalue_sum
                                       : 0.0f;

                // Map radius
                float size   = 30.0f;
                float alpha  = alphaMap[y][x];
                float radius = alpha * size;

                if (radius == 0.0f)
                {
                destR[y][x] = red[y][x];
                destG[y][x] = green[y][x];
                destB[y][x] = blue[y][x];

                break;
                }

                // Compute the width and height of an ellipse that is more width-elongated for high anisotropy
                // and more circular for low anisotropy, controlled using the eccentricity factor. Since the
                // anisotropy is in the [0, 1] range, the width factor tends to 1 as the eccentricity tends to
                // infinity and tends to infinity when the eccentricity tends to zero. This is based on the
                // equations in section "3.2. Anisotropic Kuwahara Filtering" of the paper.
                float eccentricity       = 0.5f; // TODO: Add variable for eccentricity
                float eccentricity_clamp = min (eccentricity, 0.95f);
                float eccentric_adj      = (1.0f - eccentricity_clamp) * 10.f;
                float ellipse_width_factor =
                    (eccentric_adj + anisotropy) / eccentric_adj;

                float ellipse_width  = ellipse_width_factor * radius;
                float ellipse_height = radius / ellipse_width_factor;

                // Compute the cosine and sine of the angle that the eigenvector makes with the x axis. Since the
                // eigenvector is normalized, its x and y components are the cosine and sine of the angle it
                // makes with the x axis.
                float cosine = unit_eigenvector.x;
                float sine   = unit_eigenvector.y;

                // Compute an inverse transformation matrix that represents an ellipse of the given width and
                // height and makes and an angle with the x axis of the given cosine and sine. This is an inverse
                // matrix, so it transforms the ellipse into a disk of unit radius.
                float2 inverse_ellipse_matrix_1 (
                    cosine / ellipse_width, sine / ellipse_width);
                float2 inverse_ellipse_matrix_2 (
                    -sine / ellipse_height, cosine / ellipse_height);

                // Compute the bounding box of a zero centered ellipse whose major axis is aligned with the
                // eigenvector and has the given width and height. This is based on the equations described in:
                //
                //   https://iquilezles.org/articles/ellipses/
                //
                // Notice that we only compute the upper bound, the lower bound is just negative that since the
                // ellipse is zero centered. Also notice that we take the ceiling of the bounding box, just to
                // ensure the filter window is at least 1x1.
                float2 ellipse_major_axis = ellipse_width * unit_eigenvector;
                float2 ellipse_minor_axis = float2 (
                    ellipse_height * unit_eigenvector.y * -1.f,
                    ellipse_height * unit_eigenvector.x * 1.f);

                Vec2<int> ellipse_bounds = Vec2<int> (
                    ceil (sqrt (
                        square (ellipse_major_axis.x) +
                        square (ellipse_minor_axis.x))),
                    ceil (sqrt (
                        square (ellipse_major_axis.y) +
                        square (ellipse_minor_axis.y))));

                // Compute the overlap polynomial parameters for 8-sector ellipse based on the equations in
                // section "3 Alternative Weighting Functions" of the polynomial weights paper. More on this
                // later in the code.
                const int number_of_sectors               = 8;
                float     sector_center_overlap_parameter = 2.f / radius;
                float     sector_envelope_angle =
                    ((3.f / 2.f) * PI) / number_of_sectors;
                float cross_sector_overlap_parameter =
                    (sector_center_overlap_parameter +
                     cos (sector_envelope_angle)) /
                    square (sin (sector_envelope_angle));

                // We need to compute the weighted mean of color and squared color of each of the 8 sectors of
                // the ellipse, so we declare arrays for accumulating those and initialize them in the next code
                // section.
                float4 weighted_mean_of_squared_color_of_sectors[8];
                float4 weighted_mean_of_color_of_sectors[8];
                float  sum_of_weights_of_sectors[8];

                // The center pixel (0, 0) is exempt from the main loop below for reasons that are explained in
                // the first if statement in the loop, so we need to accumulate its color, squared color, and
                // weight separately first. Luckily, the zero coordinates of the center pixel zeros out most of
                // the complex computations below, and it can easily be shown that the weight for the center
                // pixel in all sectors is simply (1 / number_of_sectors).
                float4 center_color          = float4 (
                    red[y][x], 
                    green[y][x], 
                    blue[y][x],
                    0
                );
                float4 center_color_squared  = center_color * center_color;
                float  center_weight_b       = 1.f / number_of_sectors;
                float4 weighted_center_color = center_color * center_weight_b;
                float4 weighted_center_color_squared =
                    center_color_squared * center_weight_b;

                for (int i = 0; i < number_of_sectors; i++)
                {
                weighted_mean_of_squared_color_of_sectors[i] =
                    weighted_center_color_squared;
                weighted_mean_of_color_of_sectors[i] = weighted_center_color;
                sum_of_weights_of_sectors[i]         = center_weight_b;
                }

                // Loop over the window of pixels inside the bounding box of the ellipse. However, we utilize the
                // fact that ellipses are mirror symmetric along the horizontal axis, so we reduce the window to
                // only the upper two quadrants, and compute each two mirrored pixels at the same time using the
                // same weight as an optimization.
                for (int j = 0; j <= ellipse_bounds.y; j++)
                {
                for (int i = -ellipse_bounds.x; i <= ellipse_bounds.x; i++)
                {

                    // Since we compute each two mirrored pixels at the same time, we need to also exempt the
                    // pixels whose x coordinates are negative and their y coordinates are zero, that's because
                    // those are mirrored versions of the pixels whose x coordinates are positive and their y
                    // coordinates are zero, and we don't want to compute and accumulate them twice. Moreover, we
                    // also need to exempt the center pixel with zero coordinates for the same reason, however,
                    // since the mirror of the center pixel is itself, it need to be accumulated separately,
                    // hence why we did that in the code section just before this loop.
                    if (j == 0 && i <= 0) { continue; }

                    // Map the pixels of the ellipse into a unit disk, exempting any points that are not part of
                    // the ellipse or disk.
                    float2 disk_point (
                        (inverse_ellipse_matrix_1.x * i +
                         inverse_ellipse_matrix_1.y * j),
                        (inverse_ellipse_matrix_2.x * i +
                         inverse_ellipse_matrix_2.y * j));

                    float disk_point_length_squared =
                        dot (disk_point, disk_point);
                    if (disk_point_length_squared > 1.0f) { continue; }

                    // While each pixel belongs to a single sector in the ellipse, we expand the definition of
                    // a sector a bit to also overlap with other sectors as illustrated in Figure 8 of the
                    // polynomial weights paper. So each pixel may contribute to multiple sectors, and thus we
                    // compute its weight in each of the 8 sectors.
                    float sector_weights[8];

                    // We evaluate the weighting polynomial at each of the 8 sectors by rotating the disk point
                    // by 45 degrees and evaluating the weighting polynomial at each incremental rotation. To
                    // avoid potentially expensive rotations, we utilize the fact that rotations by 90 degrees
                    // are simply swapping of the coordinates and negating the x component. We also note that
                    // since the y term of the weighting polynomial is squared, it is not affected by the sign
                    // and can be computed once for the x and once for the y coordinates. So we compute every
                    // other even-indexed 4 weights by successive 90 degree rotations as discussed.

                    float2 polynomial = (sector_center_overlap_parameter -
                                        cross_sector_overlap_parameter) *
                                            disk_point * disk_point;
                    sector_weights[0] =
                        square (max (0.f, disk_point.y + polynomial.x));
                    sector_weights[2] =
                        square (max (0.f, -disk_point.x + polynomial.y));
                    sector_weights[4] =
                        square (max (0.f, -disk_point.y + polynomial.x));
                    sector_weights[6] =
                        square (max (0.f, disk_point.x + polynomial.y));

                    // Then we rotate the disk point by 45 degrees, which is a simple expression involving a
                    // constant as can be demonstrated by applying a 45 degree rotation matrix.
                    float M_SQRT1_2 =
                        1.0f /
                        sqrt (2.0f); // M_SQRT1_2 = 0.70710678118654752440f
                    float2 rotated_disk_point =
                        M_SQRT1_2 * float2 (
                                        disk_point.x - disk_point.y,
                                        disk_point.x + disk_point.y);

                    // Finally, we compute every other odd-index 4 weights starting from the 45 degrees rotated disk point.
                    float2 rotated_polynomial =
                        (sector_center_overlap_parameter -
                        cross_sector_overlap_parameter) * rotated_disk_point *
                            rotated_disk_point;
                    sector_weights[1] = square (
                        max (0.f, rotated_disk_point.y + rotated_polynomial.x));
                    sector_weights[3] = square (max (
                        0.f, -rotated_disk_point.x + rotated_polynomial.y));
                    sector_weights[5] = square (max (
                        0.f, -rotated_disk_point.y + rotated_polynomial.x));
                    sector_weights[7] = square (
                        max (0.f, rotated_disk_point.x + rotated_polynomial.y));

                    // We compute a radial Gaussian weighting component such that pixels further away from the
                    // sector center gets attenuated, and we also divide by the sum of sector weights to
                    // normalize them, since the radial weight will eventually be multiplied to the sector weight below.
                    float sector_weights_sum =
                        sector_weights[0] + sector_weights[1] +
                        sector_weights[2] + sector_weights[3] +
                        sector_weights[4] + sector_weights[5] +
                        sector_weights[6] + sector_weights[7];
                    float radial_gaussian_weight =
                        exp (-PI * disk_point_length_squared) /
                        sector_weights_sum;

                    // Load the color of the pixel and its mirrored pixel and compute their square.
                    float4 upper_color = float4 (
                        red[hiY][hiX], green[hiY][hiX], blue[hiY][hiX], 0);
                    float4 lower_color = float4 (
                        red[lowY][lowX],
                        green[lowY][lowX],
                        blue[lowY][lowX],
                        0);
                    float4 upper_color_squared = upper_color * upper_color;
                    float4 lower_color_squared = lower_color * lower_color;

                    for (int k = 0; k < number_of_sectors; k++)
                    {
                        float weight =
                            sector_weights[k] * radial_gaussian_weight;

                        // Accumulate the pixel to each of the sectors multiplied by the sector weight.
                        int upper_index = k;
                        sum_of_weights_of_sectors[upper_index] += weight;
                        weighted_mean_of_color_of_sectors[upper_index] +=
                            upper_color * weight;
                        weighted_mean_of_squared_color_of_sectors
                            [upper_index] += upper_color_squared * weight;

                        // Accumulate the mirrored pixel to each of the sectors multiplied by the sector weight.
                        int lower_index =
                            (k + number_of_sectors / 2) % number_of_sectors;
                        sum_of_weights_of_sectors[lower_index] += weight;
                        weighted_mean_of_color_of_sectors[lower_index] +=
                            lower_color * weight;
                        weighted_mean_of_squared_color_of_sectors
                            [lower_index] += lower_color_squared * weight;
                    }
                }
                }

                // Compute the weighted sum of mean of sectors, such that sectors with lower standard deviation
                // gets more significant weight than sectors with higher standard deviation.
                float  sum_of_weights = 0.f;
                float4 weighted_sum   = float4 (0.f);
                for (int i = 0; i < number_of_sectors; i++)
                {
                weighted_mean_of_color_of_sectors[i] /=
                    sum_of_weights_of_sectors[i];
                weighted_mean_of_squared_color_of_sectors[i] /=
                    sum_of_weights_of_sectors[i];

                float4 color_mean = weighted_mean_of_color_of_sectors[i];
                float4 squared_color_mean =
                    weighted_mean_of_squared_color_of_sectors[i];
                float4 non_pos_color_var =
                    squared_color_mean - color_mean * color_mean;
                float4 color_variance = float4 (
                    fabs (non_pos_color_var.w),
                    fabs (non_pos_color_var.x),
                    fabs (non_pos_color_var.y),
                    fabs (non_pos_color_var.z));

                float standard_deviation = dot (
                    float3 (
                        sqrt (color_variance.x), sqrt (color_variance.y), sqrt (color_variance.z)),
                    float3 (1.0f));

                // Compute the sector weight based on the weight function introduced in section "3.3.1
                // Single-scale Filtering" of the multi-scale paper. Use a threshold of 0.02 to avoid zero
                // division and avoid artifacts in homogeneous regions as demonstrated in the paper.
                float sharpness            = 5.0f;
                float normalized_sharpness = sharpness * 10.0f;
                float weight =
                    1.0 /
                    pow (max (0.02f, standard_deviation), normalized_sharpness);

                sum_of_weights += weight;
                weighted_sum += color_mean * weight;
                }

                weighted_sum /= sum_of_weights;
                //dst() = weighted_sum;
                destR[y][x] = weighted_sum.x;
                destG[y][x] = weighted_sum.y;
                destB[y][x] = weighted_sum.z;
                destA[y][x] = alpha;
            }
        }
}

template <typename T>
void
helperForMultipleChannelThreading (
    int start,
    int end,
    int      width,
    Array2D<float>& tensorR,
    Array2D<float>& tensorG,
    Array2D<float>& tensorB,
    Array2D<T>& trueRed,
    Array2D<T>& trueGreen,
    Array2D<T>&  trueBlue,
    Array2D<T>&  trueAlpha)
{
        for (int y = start; y < end; y++)
        {
            cout << y << "/" << end << endl;
            int lowY = max (0, y - 1);
            int hiY  = min (end - 1, y + 1);
            for (int x = 0; x < width; x++)
            {
                int lowX = max (0, x - 1);
                int hiX  = min (width - 1, x + 1);

                // The structure tensor is encoded in a float4 using a column major storage order, as can be seen
                // in the compositor_kuwahara_anisotropic_compute_structure_tensor.glsl shader

                float dxdx = tensorR[y][x];
                float dxdy = tensorG[y][x];
                float dydy = tensorB[y][x];

                // Compute the first and second eigenvalues of the structure tensor using the equations in
                // section "3.1 Orientation and Anisotropy Estimation" of the paper.
                float eigenvalue_first_term = (dxdx + dydy) / 2.f;
                float eigenvalue_square_root_term =
                    sqrt (square (dxdx - dydy) + 4.0f * square (dxdy)) / 2.0f;
                float first_eigenvalue =
                    eigenvalue_first_term + eigenvalue_square_root_term;
                float second_eigenvalue =
                    eigenvalue_first_term - eigenvalue_square_root_term;

                // Compute the normalized eigenvector of the structure tensor oriented in direction of the
                // minimum rate of change using the equations in section "3.1 Orientation and Anisotropy
                // Estimation" of the paper.
                float2 eigenvector =
                    float2 (first_eigenvalue - dxdx, -1.0f * dxdy);
                float  eigenvector_length = length (eigenvector);
                float2 unit_eigenvector   = eigenvector_length != 0.0f
                                                ? eigenvector / eigenvector_length
                                                : float2 (1.0f);

                // Compute the amount of anisotropy using equations in section "3.1 Orientation and Anisotropy
                // Estimation" of the paper. The anisotropy ranges from 0 to 1, where 0 corresponds to isotropic
                // and 1 corresponds to entirely anisotropic regions.
                float eigenvalue_sum = first_eigenvalue + second_eigenvalue;
                float eigenvalue_difference =
                    first_eigenvalue - second_eigenvalue;
                float anisotropy = eigenvalue_sum > 0.0f
                                       ? eigenvalue_difference / eigenvalue_sum
                                       : 0.0f;
                float size       = 30.0f;
                float radius     = max (0.0f, size);
                if (radius == 0.0f) { break; }

                // Compute the width and height of an ellipse that is more width-elongated for high anisotropy
                // and more circular for low anisotropy, controlled using the eccentricity factor. Since the
                // anisotropy is in the [0, 1] range, the width factor tends to 1 as the eccentricity tends to
                // infinity and tends to infinity when the eccentricity tends to zero. This is based on the
                // equations in section "3.2. Anisotropic Kuwahara Filtering" of the paper.
                float eccentricity       = 0.5f;
                float eccentricity_clamp = min (eccentricity, 0.95f);
                float eccentric_adj      = (1.0f - eccentricity_clamp) * 10.f;
                float ellipse_width_factor =
                    (eccentric_adj + anisotropy) / eccentric_adj;

                float ellipse_width  = ellipse_width_factor * radius;
                float ellipse_height = radius / ellipse_width_factor;

                // Compute the cosine and sine of the angle that the eigenvector makes with the x axis. Since the
                // eigenvector is normalized, its x and y components are the cosine and sine of the angle it
                // makes with the x axis.
                float cosine = unit_eigenvector.x;
                float sine   = unit_eigenvector.y;

                // Compute an inverse transformation matrix that represents an ellipse of the given width and
                // height and makes and an angle with the x axis of the given cosine and sine. This is an inverse
                // matrix, so it transforms the ellipse into a disk of unit radius.
                float2 inverse_ellipse_matrix_1 (
                    cosine / ellipse_width, sine / ellipse_width);
                float2 inverse_ellipse_matrix_2 (
                    -sine / ellipse_height, cosine / ellipse_height);

                // Compute the bounding box of a zero centered ellipse whose major axis is aligned with the
                // eigenvector and has the given width and height. This is based on the equations described in:
                //
                //   https://iquilezles.org/articles/ellipses/
                //
                // Notice that we only compute the upper bound, the lower bound is just negative that since the
                // ellipse is zero centered. Also notice that we take the ceiling of the bounding box, just to
                // ensure the filter window is at least 1x1.
                float2 ellipse_major_axis = ellipse_width * unit_eigenvector;
                float2 ellipse_minor_axis = float2 (
                    ellipse_height * unit_eigenvector.y * -1.f,
                    ellipse_height * unit_eigenvector.x * 1.f);

                Vec2<int> ellipse_bounds = Vec2<int> (
                    ceil (sqrt (
                        square (ellipse_major_axis.x) +
                        square (ellipse_minor_axis.x))),
                    ceil (sqrt (
                        square (ellipse_major_axis.y) +
                        square (ellipse_minor_axis.y))));

                // Compute the overlap polynomial parameters for 8-sector ellipse based on the equations in
                // section "3 Alternative Weighting Functions" of the polynomial weights paper. More on this
                // later in the code.
                const int number_of_sectors               = 8;
                float     sector_center_overlap_parameter = 2.f / radius;
                float     sector_envelope_angle =
                    ((3.f / 2.f) * PI) / number_of_sectors;
                float cross_sector_overlap_parameter =
                    (sector_center_overlap_parameter +
                     cos (sector_envelope_angle)) /
                    square (sin (sector_envelope_angle));

                // We need to compute the weighted mean of color and squared color of each of the 8 sectors of
                // the ellipse, so we declare arrays for accumulating those and initialize them in the next code
                // section.
                float4 weighted_mean_of_squared_color_of_sectors[8];
                float4 weighted_mean_of_color_of_sectors[8];
                float  sum_of_weights_of_sectors[8];

                // The center pixel (0, 0) is exempt from the main loop below for reasons that are explained in
                // the first if statement in the loop, so we need to accumulate its color, squared color, and
                // weight separately first. Luckily, the zero coordinates of the center pixel zeros out most of
                // the complex computations below, and it can easily be shown that the weight for the center
                // pixel in all sectors is simply (1 / number_of_sectors).
                float4 center_color = float4 (
                    trueRed[y][x], trueGreen[y][x], trueBlue[y][x], trueAlpha[y][x]);
                float4 center_color_squared  = center_color * center_color;
                float  center_weight_b       = 1.f / number_of_sectors;
                float4 weighted_center_color = center_color * center_weight_b;
                float4 weighted_center_color_squared =
                    center_color_squared * center_weight_b;

                for (int i = 0; i < number_of_sectors; i++)
                {
                weighted_mean_of_squared_color_of_sectors[i] =
                    weighted_center_color_squared;
                weighted_mean_of_color_of_sectors[i] = weighted_center_color;
                sum_of_weights_of_sectors[i]         = center_weight_b;
                }

                // Loop over the window of pixels inside the bounding box of the ellipse. However, we utilize the
                // fact that ellipses are mirror symmetric along the horizontal axis, so we reduce the window to
                // only the upper two quadrants, and compute each two mirrored pixels at the same time using the
                // same weight as an optimization.
                for (int j = 0; j <= ellipse_bounds.y; j++)
                {
                for (int i = -ellipse_bounds.x; i <= ellipse_bounds.x; i++)
                {

                    // Since we compute each two mirrored pixels at the same time, we need to also exempt the
                    // pixels whose x coordinates are negative and their y coordinates are zero, that's because
                    // those are mirrored versions of the pixels whose x coordinates are positive and their y
                    // coordinates are zero, and we don't want to compute and accumulate them twice. Moreover, we
                    // also need to exempt the center pixel with zero coordinates for the same reason, however,
                    // since the mirror of the center pixel is itself, it need to be accumulated separately,
                    // hence why we did that in the code section just before this loop.
                    if (j == 0 && i <= 0) { continue; }

                    // Map the pixels of the ellipse into a unit disk, exempting any points that are not part of
                    // the ellipse or disk.
                    float2 disk_point (
                        (inverse_ellipse_matrix_1.x * i +
                         inverse_ellipse_matrix_1.y * j),
                        (inverse_ellipse_matrix_2.x * i +
                         inverse_ellipse_matrix_2.y * j));

                    float disk_point_length_squared =
                        dot (disk_point, disk_point);
                    if (disk_point_length_squared > 1.0f) { continue; }

                    // While each pixel belongs to a single sector in the ellipse, we expand the definition of
                    // a sector a bit to also overlap with other sectors as illustrated in Figure 8 of the
                    // polynomial weights paper. So each pixel may contribute to multiple sectors, and thus we
                    // compute its weight in each of the 8 sectors.
                    float sector_weights[8];

                    // We evaluate the weighting polynomial at each of the 8 sectors by rotating the disk point
                    // by 45 degrees and evaluating the weighting polynomial at each incremental rotation. To
                    // avoid potentially expensive rotations, we utilize the fact that rotations by 90 degrees
                    // are simply swapping of the coordinates and negating the x component. We also note that
                    // since the y term of the weighting polynomial is squared, it is not affected by the sign
                    // and can be computed once for the x and once for the y coordinates. So we compute every
                    // other even-indexed 4 weights by successive 90 degree rotations as discussed.

                    float2 polynomial = (sector_center_overlap_parameter -
                                         cross_sector_overlap_parameter) *
                                        disk_point * disk_point;
                    sector_weights[0] =
                        square (max (0.f, disk_point.y + polynomial.x));
                    sector_weights[2] =
                        square (max (0.f, -disk_point.x + polynomial.y));
                    sector_weights[4] =
                        square (max (0.f, -disk_point.y + polynomial.x));
                    sector_weights[6] =
                        square (max (0.f, disk_point.x + polynomial.y));

                    // Then we rotate the disk point by 45 degrees, which is a simple expression involving a
                    // constant as can be demonstrated by applying a 45 degree rotation matrix.
                    float M_SQRT1_2 =
                        1.0f /
                        sqrt (2.0f); // M_SQRT1_2 = 0.70710678118654752440f
                    float2 rotated_disk_point =
                        M_SQRT1_2 * float2 (
                                        disk_point.x - disk_point.y,
                                        disk_point.x + disk_point.y);

                    // Finally, we compute every other odd-index 4 weights starting from the 45 degrees rotated disk point.
                    float2 rotated_polynomial =
                        (sector_center_overlap_parameter -
                         cross_sector_overlap_parameter) *
                        rotated_disk_point * rotated_disk_point;
                    sector_weights[1] = square (
                        max (0.f, rotated_disk_point.y + rotated_polynomial.x));
                    sector_weights[3] = square (max (
                        0.f, -rotated_disk_point.x + rotated_polynomial.y));
                    sector_weights[5] = square (max (
                        0.f, -rotated_disk_point.y + rotated_polynomial.x));
                    sector_weights[7] = square (
                        max (0.f, rotated_disk_point.x + rotated_polynomial.y));

                    // We compute a radial Gaussian weighting component such that pixels further away from the
                    // sector center gets attenuated, and we also divide by the sum of sector weights to
                    // normalize them, since the radial weight will eventually be multiplied to the sector weight below.
                    float sector_weights_sum =
                        sector_weights[0] + sector_weights[1] +
                        sector_weights[2] + sector_weights[3] +
                        sector_weights[4] + sector_weights[5] +
                        sector_weights[6] + sector_weights[7];
                    float radial_gaussian_weight =
                        exp (-PI * disk_point_length_squared) /
                        sector_weights_sum;

                    // Load the color of the pixel and its mirrored pixel and compute their square.
                    float4 upper_color = float4 (
                        trueRed[hiY][hiX],
                        trueGreen[hiY][hiX],
                        trueBlue[hiY][hiX],
                        trueAlpha[hiY][hiX]);
                    float4 lower_color = float4 (
                        trueRed[lowY][lowX],
                        trueGreen[lowY][lowX],
                        trueBlue[lowY][lowX],
                        trueAlpha[lowY][lowX]);
                    float4 upper_color_squared = upper_color * upper_color;
                    float4 lower_color_squared = lower_color * lower_color;

                    for (int k = 0; k < number_of_sectors; k++)
                    {
                        float weight =
                            sector_weights[k] * radial_gaussian_weight;

                        // Accumulate the pixel to each of the sectors multiplied by the sector weight.
                        int upper_index = k;
                        sum_of_weights_of_sectors[upper_index] += weight;
                        weighted_mean_of_color_of_sectors[upper_index] +=
                            upper_color * weight;
                        weighted_mean_of_squared_color_of_sectors
                            [upper_index] += upper_color_squared * weight;

                        // Accumulate the mirrored pixel to each of the sectors multiplied by the sector weight.
                        int lower_index =
                            (k + number_of_sectors / 2) % number_of_sectors;
                        sum_of_weights_of_sectors[lower_index] += weight;
                        weighted_mean_of_color_of_sectors[lower_index] +=
                            lower_color * weight;
                        weighted_mean_of_squared_color_of_sectors
                            [lower_index] += lower_color_squared * weight;
                    }
                }
                }

                // Compute the weighted sum of mean of sectors, such that sectors with lower standard deviation
                // gets more significant weight than sectors with higher standard deviation.
                float  sum_of_weights = 0.f;
                float4 weighted_sum   = float4 (0.f);
                for (int i = 0; i < number_of_sectors; i++)
                {
                weighted_mean_of_color_of_sectors[i] /=
                    sum_of_weights_of_sectors[i];
                weighted_mean_of_squared_color_of_sectors[i] /=
                    sum_of_weights_of_sectors[i];

                float4 color_mean = weighted_mean_of_color_of_sectors[i];
                float4 squared_color_mean =
                    weighted_mean_of_squared_color_of_sectors[i];
                float4 non_pos_color_var =
                    squared_color_mean - color_mean * color_mean;
                float4 color_variance = float4 (
                    fabs (non_pos_color_var.w),
                    fabs (non_pos_color_var.x),
                    fabs (non_pos_color_var.y),
                    fabs (non_pos_color_var.z));

                float standard_deviation = dot (
                    float3 (
                        sqrt (color_variance.x),
                        sqrt (color_variance.y),
                        sqrt (color_variance.z)),
                    float3 (1.0f));

                // Compute the sector weight based on the weight function introduced in section "3.3.1
                // Single-scale Filtering" of the multi-scale paper. Use a threshold of 0.02 to avoid zero
                // division and avoid artifacts in homogeneous regions as demonstrated in the paper.
                float sharpness            = 5.0f;
                float normalized_sharpness = sharpness * 10.0f;
                float weight =
                    1.0 /
                    pow (max (0.02f, standard_deviation), normalized_sharpness);

                sum_of_weights += weight;
                weighted_sum += color_mean * weight;
                }

                weighted_sum /= sum_of_weights;
                trueRed[y][x]   = weighted_sum.x;
                trueGreen[y][x] = weighted_sum.y;
                trueBlue[y][x]  = weighted_sum.z;
                trueAlpha[y][x] = center_color.w;
            }
        }
}

template <typename T>
void
modifyChannels (list<Array2D<T>>& channels, ChannelList& headderChannels)
{
    //
    // Dummy code modifying each pixel by incrementing every channel by a given delta.
    //
    ChannelList::ConstIterator header = headderChannels.begin ();
    
    Array2D<T>*                    red   = (Array2D<T>*) nullptr;
    Array2D<T>*                    green = (Array2D<T>*) nullptr;
    Array2D<T>*                    blue  = (Array2D<T>*) nullptr;
    Array2D<T>*                    alpha  = (Array2D<T>*) nullptr;
    cout << "here" << endl;
    for (list<Array2D<T>>::iterator i = channels.begin (); i != channels.end (); i++)
    {
        cout << "iter" << endl;
        cout << header.name () << " at: " << &*i << endl;
        if (std::strncmp (header.name (), "R", 1) == 0) { 
            red = (Array2D<T>*) &*i;
        }
        if (std::strncmp (header.name (), "G", 1) == 0)
        {
            green = (Array2D<T>*) &*i;
        }
        if (std::strncmp (header.name (), "B", 1) == 0)
        {
            blue = (Array2D<T>*) &*i;
        }
        if (std::strncmp (header.name (), "A", 1) == 0)
        {
            alpha = (Array2D<T>*) &*i;
        }


        
        header++;
    }
    cout << red << endl;
    cout << green << endl;
    cout << blue << endl;
    cout << alpha << endl;

    const float corner_weight = 0.182f;
    const float center_weight = 1.0f - 2.0f * corner_weight;
    if (red != nullptr && blue != nullptr && green != nullptr)
    {
        cout << "Here" << endl;
        int height = min (
            min (red->height (), blue->height ()),
            min (green->height (), alpha->height()));
        int width = min (
            min (red->width (), blue->width ()), min (green->width (), alpha->width()));
        cout << width << endl;
        cout << height << endl;
        Array2D<T>& trueRed = *red;
        Array2D<T>& trueGreen = *green;
        Array2D<T>& trueBlue = *blue;
        Array2D<T>& trueAlpha = *alpha;

        Array2D<float> tensorR = Array2D<float> (width, height);
        Array2D<float> tensorG = Array2D<float> (width, height);
        Array2D<float> tensorB = Array2D<float> (width, height);

        
        // Array2D<float> mapR = Array2D<float> (width, height);
        // Array2D<float> mapG = Array2D<float> (width, height);
        // Array2D<float> mapB = Array2D<float> (width, height);
        // Array2D<float> mapA = Array2D<float> (width, height);

        cout << "Starting Tensor" << endl;
        getTensor (
            trueRed, trueGreen, trueBlue, trueAlpha, tensorR, tensorG, tensorB);
        // cout << "Tensor Finished, Starting Map" << endl;
        // getMap (tensorR, tensorG, tensorB, trueAlpha, mapR, mapG, mapB, mapA);
        cout << "Starting Main" << endl;
        std::vector<std::thread> ThreadVector;
        int                      threadCount = 8;
        int                      groupSize   = height / threadCount;
        for (int i = 0; i < threadCount; i++) {
            int start = i * groupSize;
            int end;
            if (i == threadCount - 1) { end = height;
            } else { end = (i + 1) * groupSize;
            }

                std::thread temp (
                    helperForMultipleChannelThreading<T>,
                    start,
                    end,
                    width,
                    std::ref (tensorR),
                    std::ref (tensorG),
                    std::ref (tensorB),
                    std::ref (trueRed),
                    std::ref (trueGreen),
                    std::ref (trueBlue),
                    std::ref (trueAlpha)
                  );
                ThreadVector.push_back (std::move (temp));

                cout << "Thread started from " << start << " to " << end
                     << endl;
            
        }
        
        for (auto& i: ThreadVector)
        {
            i.join ();
        }
    }

    
    
}

template <typename T>
void
modifyDeepChannels (
    Array2D<uint32_t>& sampleCount, list<Array2D<T*>>& channels, T delta)
{
    //
    // Dummy code modifying each deep pixel by incrementing every sample of each channel by a given delta.
    //

    for (auto i = channels.begin (); i != channels.end (); i++)
    {
        Array2D<T*>& channel = *i;

        for (int y = 0; y < channel.height (); y++)
        {
            for (int x = 0; x < channel.width (); x++)
            {
                uint32_t count = sampleCount[y][x];
                for (uint32_t j = 0; j < count; j++)
                    channel[y][x][j] += delta;
            }
        }
    }
}

void
modifyMultipart (const char* input, const char* output)
{
    //
    // Read all channels from a multi-part file, modify each pixel value, write the modified data as a multi-part file.
    // The parts in the file can be scanline- or tile-based, either flat or deep.
    // Every channel of the input file gets modified.
    //
    MultiPartInputFile inputFile (input);
    // MultiPartInputFile inputFile ("rgba3.exr");

    std::vector<Header> headers (inputFile.parts ());

    for (int i = 0; i < inputFile.parts (); i++)
    {
        headers[i] = inputFile.header (i);
    }

    MultiPartOutputFile outputFile (
        output, headers.data (), (int) headers.size ());

    for (int i = 0; i < inputFile.parts (); i++)
    {
        Header& header = headers[i];
        ChannelList& channel_list = header.channels ();


        const string& type = header.type ();

        if (type == SCANLINEIMAGE || type == TILEDIMAGE)
        {
            list<Array2D<uint32_t>> intChannels;
            list<Array2D<half>>     halfChannels;
            list<Array2D<float>>    floatChannels;

            FrameBuffer frameBuffer = setupFramebuffer (
                header, intChannels, halfChannels, floatChannels);
            if (type == SCANLINEIMAGE)
            {
                InputPart inputPart (inputFile, i);
                inputPart.setFrameBuffer (frameBuffer);
                inputPart.readPixels (
                    header.dataWindow ().min.y, header.dataWindow ().max.y);
            }
            else
            {
                TiledInputPart inputPart (inputFile, i);
                inputPart.setFrameBuffer (frameBuffer);
                inputPart.readTiles (
                    0,
                    inputPart.numXTiles () - 1,
                    0,
                    inputPart.numYTiles () - 1);
            }

            modifyChannels < uint32_t>(intChannels, channel_list);
            modifyChannels<half> (halfChannels, channel_list);
            modifyChannels<float> (floatChannels, channel_list);

            if (type == SCANLINEIMAGE)
            {
                Box2i      dataWindow = header.dataWindow ();
                OutputPart outputPart (outputFile, i);
                outputPart.setFrameBuffer (frameBuffer);
                outputPart.writePixels (
                    dataWindow.max.y - dataWindow.min.y + 1);
            }
            else
            {
                TiledOutputPart outputPart (outputFile, i);
                outputPart.setFrameBuffer (frameBuffer);
                outputPart.writeTiles (
                    0,
                    outputPart.numXTiles () - 1,
                    0,
                    outputPart.numYTiles () - 1);
            }
        }
    }
    
}

void
multipartExamples ()
{
    // Read multiple single-part files and write them out as a single multi-part file.
  //  combineFiles ();

    // Read all parts from a multi-part file, modify each channel of every pixel by incrementing its value, write out as a multi-part file.
    modifyMultipart ("rgba3.exr", "test_output-rgba-multiple-runs-1.exr");
    modifyMultipart ("test_output-rgba-multiple-runs-1.exr", "test_output-rgba-multiple-runs-2.exr");
    modifyMultipart (
        "test_output-rgba-multiple-runs-2.exr", "test_output-rgba-multiple-runs-3.exr");
    // Read a multi-part file and write out as multiple single-part files.
    // splitFiles ();
}
