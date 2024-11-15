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
#include <iostream>
using namespace Windows::Foundation::Numerics;
using namespace IMF;
using namespace std;
using namespace IMATH_NAMESPACE;

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
        cout << header.name () << endl;
        if (std::strncmp (header.name (), "R", 1)) { 
            red = (Array2D<T>*) &*i;
        }
        if (std::strncmp (header.name (), "G", 1))
        {
            green = (Array2D<T>*) &*i;
        }
        if (std::strncmp (header.name (), "B", 1))
        {
            blue = (Array2D<T>*) &*i;
        }
        if (std::strncmp (header.name (), "A", 1))
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
        cout << red[50][50] << endl;
        cout << red[0][0] << endl;
        cout << red[1079][1919] << endl;

        for (int y = 0; y < height; y++)
        {
        for (int x = 0; x < width; x++)
            {
                int lowX = max (0, x - 1);
                int hiX  = min (width - 1, x + 1);
                int lowY = max (0, y - 1);
                int hiY  = min (height - 1, y + 1);

                Array2D<T>& trueRed = *red;
                Array2D<T>& trueGreen = *green;
                Array2D<T>& trueBlue = *blue;
                Array2D<T>& trueAlpha = *alpha;


                float red1 = (trueRed[hiY][lowX]);
                float red2 = (trueRed[y][lowX]);
                float red3 = (trueRed[lowY][lowX]);
                float red4 = (trueRed[hiY][hiX]);
                float red5 = (trueRed[y][hiX]);
                float red6 = (trueRed[lowY][hiX]);
                float r = red1 * -corner_weight +
                      red2 * -center_weight +
                      red3 * -corner_weight +
                      red4 * corner_weight +
                      red5 * center_weight +
                      red6 * corner_weight;

                float green1 = (trueGreen[hiY][lowX]);
                float green2 = (trueGreen[y][lowX]);
                float green3 = (trueGreen[lowY][lowX]);
                float green4 = (trueGreen[hiY][hiX]);
                float green5 = (trueGreen[y][hiX]);
                float green6 = (trueGreen[lowY][hiX]);
                float g = green1 * -corner_weight + green2 * -center_weight +
                          green3 * -corner_weight + green4 * corner_weight +
                          green5 * center_weight + green6 * corner_weight;

                
                float blue1 = (trueBlue[hiY][lowX]);
                float blue2 = (trueBlue[y][lowX]);
                float blue3 = (trueBlue[lowY][lowX]);
                float blue4 = (trueBlue[hiY][hiX]);
                float blue5 = (trueBlue[y][hiX]);
                float blue6 = (trueBlue[lowY][hiX]);
                float b     = blue1 * -corner_weight + blue2 * -center_weight +
                          blue3 * -corner_weight + blue4 * corner_weight +
                          blue5 * center_weight + blue6 * corner_weight;

                
                float alpha1 = (trueAlpha[hiY][lowX]);
                float alpha2 = (trueAlpha[y][lowX]);
                float alpha3 = (trueAlpha[lowY][lowX]);
                float alpha4 = (trueAlpha[hiY][hiX]);
                float alpha5 = (trueAlpha[y][hiX]);
                float alpha6 = (trueAlpha[lowY][hiX]);
                float a = alpha1 * -corner_weight + alpha2 * -center_weight +
                          alpha3 * -corner_weight + alpha4 * corner_weight +
                          alpha5 * center_weight + alpha6 * corner_weight;
         
            float4 x_partial_derivative = float4 (r, g, b, a);

            float yR = trueRed[hiY][lowX] * corner_weight +
                       trueRed[hiY][x] * center_weight +
                       trueRed[hiY][hiX] * corner_weight +
                       trueRed[lowY][lowX] * -corner_weight +
                       trueRed[lowY][x] * -center_weight +
                       trueRed[lowY][hiX] * -corner_weight;

            float yG = trueGreen[hiY][lowX] * corner_weight +
                       trueGreen[hiY][x] * center_weight +
                       trueGreen[hiY][hiX] * corner_weight +
                       trueGreen[lowY][lowX] * -corner_weight +
                       trueGreen[lowY][x] * -center_weight +
                       trueGreen[lowY][hiX] * -corner_weight;

            float yB = trueBlue[hiY][lowX] * corner_weight +
                       trueBlue[hiY][x] * center_weight +
                       trueBlue[hiY][hiX] * corner_weight +
                       trueBlue[lowY][lowX] * -corner_weight +
                       trueBlue[lowY][x] * -center_weight +
                       trueBlue[lowY][hiX] * -corner_weight;

            float yA = trueAlpha[hiY][lowX] * corner_weight +
                       trueAlpha[hiY][x] * center_weight +
                       trueAlpha[hiY][hiX] * corner_weight +
                       trueAlpha[lowY][lowX] * -corner_weight +
                       trueAlpha[lowY][x] * -center_weight +
                       trueAlpha[lowY][hiX] * -corner_weight;

            float4 y_partial_derivative = float4(yR, yG, yB, yA);

            float dxdx = dot (x_partial_derivative, x_partial_derivative);
            float dxdy = dot (x_partial_derivative, y_partial_derivative);
            float dydy = dot (y_partial_derivative, y_partial_derivative);
            // trueRed[y][x]   = (T) dxdx;
            // trueGreen[y][x] = (T) dxdy;
            // trueBlue[y][x]  = (T) dydy;
            trueRed[y][x] = 0;
            trueGreen[y][x] = 0;
            trueBlue[y][x];
            }
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
modifyMultipart ()
{
    //
    // Read all channels from a multi-part file, modify each pixel value, write the modified data as a multi-part file.
    // The parts in the file can be scanline- or tile-based, either flat or deep.
    // Every channel of the input file gets modified.
    //
    MultiPartInputFile inputFile ("BG_depthInfo.exr");

    std::vector<Header> headers (inputFile.parts ());

    for (int i = 0; i < inputFile.parts (); i++)
    {
        headers[i] = inputFile.header (i);
    }

    MultiPartOutputFile outputFile (
        "test_output.exr", headers.data (), (int) headers.size ());

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
   modifyMultipart ();

    // Read a multi-part file and write out as multiple single-part files.
    // splitFiles ();
}
