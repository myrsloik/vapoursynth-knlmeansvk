/*
 * This file is part of KNLMeansVk,
 * Copyright(C) 2015-2020 Edoardo Brunetti (the KNLMeansCL implementation this is ported from),
 * Copyright(C) 2026 the KNLMeansVk contributors.
 *
 * KNLMeansVk is free software: you can redistribute it and/or modify it under the terms of
 * the GNU General Public License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * Non-local means denoising on the VapourSynth Vulkan GPU API (API 4.3), built on the
 * core's declaration driver; gpufilter.h is copied beside this file, as its header
 * instructs. The kernels in shader.comp are a port of the OpenCL kernels and keep their
 * decomposition: per-offset pixel distances, a separable box filter turning them into
 * patch distances, symmetric pair accumulation, then one weighted average.
 *
 * What the core replaces from the OpenCL implementation: device enumeration and selection
 * (device_type/device_id/ocl_* are gone, the core owns one device), the command queue and
 * per-instance image pool (exec contexts and per-frame scratch, so the filter runs
 * fmParallel instead of fmParallelRequests), and clFinish (producer pairs; nothing here
 * waits on the host). Frames arrive and leave GPU resident, so a chain of GPU filters
 * never crosses the PCIe bus around this one. The OpenCL image-format modes collapse: one
 * normalize-by-maxval path replaces mode_9_to_15bits and the 10-10-10 special case, and
 * 'info' is gone with the OpenCL platform text it printed.
 *
 * One behavioural repair: the OpenCL plugin read stale stack layers for the future
 * temporal side during the last 'd' frames of a clip. Here the search window shrinks
 * symmetrically at both ends instead, matching the original's start-of-clip behaviour.
 *
 * Built by meson (see ../meson.build), or by hand with a compiler that has C23 #embed:
 *   clang-cl /LD /MD /O2 /EHsc /std:c++20 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS ^
 *     knlmvulkan.cpp /I<this dir> /I<vapoursynth include> /I<vulkan sdk include> ^
 *     /Fe:knlmvulkan.dll
 */

#define VS_USE_API_43
#include "VapourSynth4.h"
#include "VSHelper4.h"
#include "VSVulkan4.h"

#include "gpufilter.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#ifndef KNLMVK_MAJOR_VERSION
#define KNLMVK_MAJOR_VERSION 2
#endif
#ifndef KNLMVK_MINOR_VERSION
#define KNLMVK_MINOR_VERSION 0
#endif

namespace {

/* The kernel source ships inside the binary. #embed is the direct route and needs no build
   step, but it is C23 (clang 19+, gcc 15+, not MSVC), so the build also generates
   shader_comp.h -- a raw string literal -- and that is used wherever #embed is missing.
   Define KNLMVK_NO_EMBED to force the generated header even where #embed exists, which is
   how the fallback gets exercised on a compiler that would not otherwise take it. */
#if defined(__has_embed) && !defined(KNLMVK_NO_EMBED)
#  if __has_embed("shader.comp")
#    define KNLMVK_HAVE_EMBED 1
#  endif
#endif

#ifdef KNLMVK_HAVE_EMBED
const char knlmGlsl[] = {
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wc23-extensions"
#embed "shader.comp"
#  pragma clang diagnostic pop
    , '\0'
};
#else
#  include "shader_comp.h"
#endif

/* Kernel ids, matching KNLM_KERNEL in shader.comp and the program list order. */
enum Kernel { kZero, kPack, kDistance, kHBox, kVBox, kAccumulate, kFinish, kNumKernels };

/* Must mirror the push_constant block in shader.comp field for field. */
struct KPush {
    int32_t width, height;
    int32_t scrStride;
    int32_t dstStride;
    int32_t srcStride0, srcStride1, srcStride2;
    int32_t qx, qy, qk;
    int32_t layer;
    int32_t numSlots;
    uint32_t gate;
};

/* One entry per pass, consulted by fillPush. */
struct PMeta {
    Kernel kind = kZero;
    int qx = 0, qy = 0, qk = 0;
    int layer = 0; /* pack: stack layer; finish: channel */
    int slots = 1;
};

std::string composeKernel(int kernel, int channels, const VSVideoFormat &fmt) {
    const bool isFloat = fmt.sampleType == stFloat;
    const bool isHalf = isFloat && fmt.bytesPerSample == 2;
    const char *sampleType = isFloat ? (isHalf ? "float16_t" : "float")
                                     : (fmt.bytesPerSample == 1 ? "uint8_t" : "uint16_t");
    std::string s =
        "#version 460\n"
        "#extension GL_EXT_shader_8bit_storage : require\n"
        "#extension GL_EXT_shader_16bit_storage : require\n"
        "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n"
        "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
    if (isHalf)
        s += "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
    s += "#define KNLM_KERNEL " + std::to_string(kernel) + "\n" +
        "#define KNLM_CHANNELS " + std::to_string(channels) + "\n" +
        "#define KNLM_FMT_FLOAT " + (isFloat ? "1" : "0") + "\n" +
        "#define SAMPLE_T " + sampleType + "\n";
    s += knlmGlsl;
    return s;
}

enum RefMode { rmLuma = 0, rmChroma = 1, rmYUV = 2, rmRGB = 3 };

static void VS_CC KNLMeansCreate(const VSMap *in, VSMap *out, void *, VSCore *core, const VSAPI *vsapi) noexcept {
    VSNode *node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    const VSVideoInfo *vi = vsapi->getVideoInfo(node);
    VSNode *rclip = nullptr;
    int err = 0;

    auto fail = [&](const std::string &msg) {
        vsapi->mapSetError(out, ("KNLMeans: " + msg).c_str());
        vsapi->freeNode(node);
        if (rclip)
            vsapi->freeNode(rclip);
    };

    if (!vsh::isConstantVideoFormat(vi))
        return fail("only constant format input supported");
    const VSVideoFormat &fmt = vi->format;
    if (fmt.colorFamily != cfGray && fmt.colorFamily != cfYUV && fmt.colorFamily != cfRGB)
        return fail("video format not supported");
    if (fmt.sampleType == stInteger && (fmt.bitsPerSample < 8 || fmt.bitsPerSample > 16))
        return fail("8-16 bit integer, half and single precision float input supported");
    if (fmt.sampleType == stFloat && fmt.bitsPerSample != 16 && fmt.bitsPerSample != 32)
        return fail("8-16 bit integer, half and single precision float input supported");

    rclip = vsapi->mapGetNode(in, "rclip", 0, &err);
    if (err)
        rclip = nullptr;
    if (rclip) {
        const VSVideoInfo *rvi = vsapi->getVideoInfo(rclip);
        if (rvi->width != vi->width || rvi->height != vi->height ||
            rvi->numFrames != vi->numFrames || !vsh::isSameVideoFormat(&rvi->format, &fmt))
            return fail("'rclip' does not match the source clip");
    }

    int dTmp = static_cast<int>(vsapi->mapGetInt(in, "d", 0, &err));
    if (err)
        dTmp = 1;
    int aTmp = static_cast<int>(vsapi->mapGetInt(in, "a", 0, &err));
    if (err)
        aTmp = 2;
    int sTmp = static_cast<int>(vsapi->mapGetInt(in, "s", 0, &err));
    if (err)
        sTmp = 4;
    double h = vsapi->mapGetFloat(in, "h", 0, &err);
    if (err)
        h = 1.2;
    const char *channels = vsapi->mapGetData(in, "channels", 0, &err);
    if (err)
        channels = "AUTO";
    int wmode = static_cast<int>(vsapi->mapGetInt(in, "wmode", 0, &err));
    if (err)
        wmode = 0;
    double wref = vsapi->mapGetFloat(in, "wref", 0, &err);
    if (err)
        wref = 1.0;

    if (dTmp < 0)
        return fail("'d' must be greater than or equal to 0");
    if (aTmp < 1)
        return fail("'a' must be greater than or equal to 1");
    if (sTmp < 0 || sTmp > 8)
        return fail("'s' must be in range [0, 8]");
    if (h <= 0.0)
        return fail("'h' must be greater than 0");
    if (wmode < 0 || wmode > 3)
        return fail("'wmode' must be in range [0, 3]");
    if (wref < 0.0)
        return fail("'wref' must be greater than or equal to 0");

    auto chanIs = [&](const char *what) {
        const char *a = channels, *b = what;
        for (; *a && *b; a++, b++)
            if (std::toupper(static_cast<unsigned char>(*a)) != *b)
                return false;
        return !*a && !*b;
    };

    /* Which channels are filtered jointly, and which output planes that covers. */
    RefMode refMode;
    int numChannels;
    bool process[3] = { false, false, false };
    if (chanIs("YUV")) {
        if (fmt.colorFamily != cfYUV)
            return fail("'channels' must be 'YUV', 'Y', or 'UV' with YUV color space");
        if (fmt.subSamplingW != 0 || fmt.subSamplingH != 0)
            return fail("'channels' = 'YUV' requires 4:4:4 YUV format");
        refMode = rmYUV;
        numChannels = 3;
        process[0] = process[1] = process[2] = true;
    } else if (chanIs("UV")) {
        if (fmt.colorFamily != cfYUV)
            return fail("'channels' must be 'YUV', 'Y', or 'UV' with YUV color space");
        refMode = rmChroma;
        numChannels = 2;
        process[1] = process[2] = true;
    } else if (chanIs("RGB")) {
        if (fmt.colorFamily != cfRGB)
            return fail("'channels' must be 'RGB' with RGB color space");
        refMode = rmRGB;
        numChannels = 3;
        process[0] = process[1] = process[2] = true;
    } else if (chanIs("Y")) {
        if (fmt.colorFamily == cfRGB)
            return fail("'channels' must be 'RGB' with RGB color space");
        refMode = rmLuma;
        numChannels = 1;
        process[0] = true;
    } else if (chanIs("AUTO")) {
        if (fmt.colorFamily == cfRGB) {
            refMode = rmRGB;
            numChannels = 3;
            process[0] = process[1] = process[2] = true;
        } else {
            refMode = rmLuma;
            numChannels = 1;
            process[0] = true;
        }
    } else {
        return fail("'channels' must be 'YUV', 'Y', 'UV', 'RGB' or 'AUTO'");
    }

    /* The planes filtered together; they always share dimensions, and the first one is
       where every all-scratch pass runs. */
    int procPlanes[3] = {}, numProc = 0;
    for (int p = 0; p < fmt.numPlanes; p++)
        if (process[p])
            procPlanes[numProc++] = p;
    const int gate = procPlanes[0];
    const int procW = gate ? vi->width >> fmt.subSamplingW : vi->width;
    const int procH = gate ? vi->height >> fmt.subSamplingH : vi->height;

    const int T = 2 * dTmp + 1;
    const bool temporal = dTmp > 0;
    const int C = numChannels;

    /* Scratch rows use their own stride: 64 floats keeps rows 256-byte aligned and
       independent of whatever stride the output plane's sample size produces. */
    const int scrStride = (procW + 63) & ~63;
    const VkDeviceSize planeBytes = static_cast<VkDeviceSize>(scrStride) * procH * sizeof(float);

    vsgpu::FilterDesc desc;
    desc.vi = *vi;
    desc.nodes.push_back(node);
    if (rclip)
        desc.nodes.push_back(rclip);
    const bool haveRef = rclip != nullptr;
    node = nullptr;
    rclip = nullptr; /* owned by desc from here, consumed by createFilter either way */

    for (int p = 0; p < 3; p++)
        desc.process[p] = process[p];

    /* Scratch slots: the clip stack, the rclip stack when given (distances read it, the
       averages always read the clip), the two-slot weight ping-pong, the accumulators. */
    const int slotU1a = 0;
    const int slotU1b = haveRef ? 1 : -1;
    const int slotU4a = haveRef ? 2 : 1;
    const int slotU4b = slotU4a + 1;
    const int slotU2 = slotU4b + 1;
    const int slotU5 = slotU2 + 1;
    desc.scratchCount = slotU5 + 1;
    desc.scratchDefs.resize(static_cast<size_t>(desc.scratchCount));
    desc.scratchDefs[slotU1a] = { static_cast<VkDeviceSize>(T) * C * planeBytes, 0 };
    if (haveRef)
        desc.scratchDefs[slotU1b] = { static_cast<VkDeviceSize>(T) * C * planeBytes, 0 };
    desc.scratchDefs[slotU4a] = { 2 * planeBytes, 0 };
    desc.scratchDefs[slotU4b] = { 2 * planeBytes, 0 };
    desc.scratchDefs[slotU2] = { static_cast<VkDeviceSize>(C + 1) * planeBytes, 0 };
    desc.scratchDefs[slotU5] = { planeBytes, 0 };

    /* Programs: the same source text specialized per kernel; parameters that are values
       rather than structure ride in as specialization constants, so instances differing
       only in parameters share every cached text. */
    const double sSize = static_cast<double>(2 * sTmp + 1) * (2 * sTmp + 1);
    const float h2InvNorm = static_cast<float>(255.0 * 255.0 / (3.0 * h * h * sSize));
    const float maxVal = fmt.sampleType == stInteger
        ? static_cast<float>((1 << fmt.bitsPerSample) - 1) : 1.0f;
    const int storageCounts[kNumKernels] = { 2, C + 1, 2, 2, 2, 4, 4 };
    for (int kern = 0; kern < kNumKernels; kern++) {
        vsgpu::Program prog;
        prog.glsl = composeKernel(kern, C, fmt);
        prog.storageBufferCount = storageCounts[kern];
        prog.pushConstantBytes = sizeof(KPush);
        prog.localSizeX = 16;
        prog.localSizeY = 16;
        struct SpecData {
            int32_t refMode, wmode, patch, radius;
            float h2, wref, denorm;
        } spec = { refMode, wmode, sTmp, dTmp, h2InvNorm, static_cast<float>(wref), maxVal };
        prog.specData.assign(reinterpret_cast<const uint8_t *>(&spec),
            reinterpret_cast<const uint8_t *>(&spec) + sizeof(spec));
        for (uint32_t i = 0; i < 7; i++)
            prog.specEntries.push_back({ i, i * 4, 4 });
        desc.programs.push_back(std::move(prog));
    }

    auto gateHeavy = [&](vsgpu::Pass &pass) {
        for (int p = 0; p < 3; p++)
            pass.planes[p] = (p == gate);
    };

    std::vector<PMeta> meta;

    /* Zero the accumulators. */
    {
        vsgpu::Pass pass;
        pass.program = kZero;
        pass.bindings.push_back(vsgpu::Operand::scratch(slotU2));
        pass.bindings.push_back(vsgpu::Operand::scratch(slotU5));
        gateHeavy(pass);
        desc.passes.push_back(std::move(pass));
        meta.push_back({ kZero });
    }

    /* Pack the temporal stacks; layers write disjoint ranges, so no barriers between. */
    for (int stack = 0; stack < (haveRef ? 2 : 1); stack++) {
        for (int tap = 0; tap < T; tap++) {
            vsgpu::Pass pass;
            pass.program = kPack;
            for (int c = 0; c < C; c++)
                pass.bindings.push_back(vsgpu::Operand::sourcePlane(procPlanes[c], stack, tap - dTmp));
            pass.bindings.push_back(vsgpu::Operand::scratch(stack ? slotU1b : slotU1a));
            pass.independent = true;
            gateHeavy(pass);
            desc.passes.push_back(std::move(pass));
            PMeta m;
            m.kind = kPack;
            m.layer = tap;
            meta.push_back(m);
        }
    }

    /* The search loop: one distance/box/box/accumulate round per half-window offset,
       exactly the OpenCL enqueue order (which fixes the float summation order too). */
    const int distStack = haveRef ? slotU1b : slotU1a;
    const int side = 2 * aTmp + 1, area = side * side;
    for (int k = -dTmp; k <= 0; k++) {
        for (int j = -aTmp; j <= aTmp; j++) {
            for (int i = -aTmp; i <= aTmp; i++) {
                if (k * area + j * side + i >= 0)
                    continue;
                const int slots = k ? 2 : 1;
                PMeta m;
                m.qx = i;
                m.qy = j;
                m.qk = k;
                m.slots = slots;
                {
                    vsgpu::Pass pass;
                    pass.program = kDistance;
                    pass.bindings.push_back(vsgpu::Operand::scratch(distStack));
                    pass.bindings.push_back(vsgpu::Operand::scratch(slotU4a));
                    gateHeavy(pass);
                    desc.passes.push_back(std::move(pass));
                    m.kind = kDistance;
                    meta.push_back(m);
                }
                {
                    vsgpu::Pass pass;
                    pass.program = kHBox;
                    pass.bindings.push_back(vsgpu::Operand::scratch(slotU4a));
                    pass.bindings.push_back(vsgpu::Operand::scratch(slotU4b));
                    gateHeavy(pass);
                    desc.passes.push_back(std::move(pass));
                    m.kind = kHBox;
                    meta.push_back(m);
                }
                {
                    vsgpu::Pass pass;
                    pass.program = kVBox;
                    pass.bindings.push_back(vsgpu::Operand::scratch(slotU4b));
                    pass.bindings.push_back(vsgpu::Operand::scratch(slotU4a));
                    gateHeavy(pass);
                    desc.passes.push_back(std::move(pass));
                    m.kind = kVBox;
                    meta.push_back(m);
                }
                {
                    vsgpu::Pass pass;
                    pass.program = kAccumulate;
                    pass.bindings.push_back(vsgpu::Operand::scratch(slotU1a));
                    pass.bindings.push_back(vsgpu::Operand::scratch(slotU4a));
                    pass.bindings.push_back(vsgpu::Operand::scratch(slotU2));
                    pass.bindings.push_back(vsgpu::Operand::scratch(slotU5));
                    gateHeavy(pass);
                    desc.passes.push_back(std::move(pass));
                    m.kind = kAccumulate;
                    meta.push_back(m);
                }
            }
        }
    }

    /* One weighted average per processed plane. */
    for (int c = 0; c < numProc; c++) {
        vsgpu::Pass pass;
        pass.program = kFinish;
        pass.bindings.push_back(vsgpu::Operand::scratch(slotU1a));
        pass.bindings.push_back(vsgpu::Operand::scratch(slotU2));
        pass.bindings.push_back(vsgpu::Operand::scratch(slotU5));
        pass.bindings.push_back(vsgpu::Operand::output());
        for (int p = 0; p < 3; p++)
            pass.planes[p] = (p == procPlanes[c]);
        desc.passes.push_back(std::move(pass));
        PMeta m;
        m.kind = kFinish;
        m.layer = c;
        meta.push_back(m);
    }

    /* The valid temporal radius for this frame: how far the clip actually extends on both
       sides, so the window shrinks symmetrically at the clip edges instead of reading
       replicated frames (start of clip) or stale data (the OpenCL end-of-clip bug). */
    const int numFrames = vi->numFrames;
    desc.frameParamCount = 1;
    desc.prepareFrame = [dTmp, numFrames](int n, const VSFrame *const *, int, const VSAPI *,
        uint32_t *params, std::string &) {
        params[0] = static_cast<uint32_t>(std::min({ dTmp, n, numFrames - 1 - n }));
        return true;
    };
    desc.mapFrame = [numFrames](int n, int, int frameOffset) {
        return std::clamp(n + frameOffset, 0, numFrames - 1);
    };

    const int scrStrideV = scrStride;
    desc.fillPush = [meta, scrStrideV](const vsgpu::PassInfo &info, void *pushData) {
        const PMeta &m = meta[info.pass];
        KPush push = {};
        push.width = static_cast<int32_t>(info.width);
        push.height = static_cast<int32_t>(info.height);
        push.scrStride = scrStrideV;
        push.qx = m.qx;
        push.qy = m.qy;
        push.qk = m.qk;
        push.layer = m.layer;
        push.numSlots = m.slots;
        push.gate = 1;
        switch (m.kind) {
        case kPack:
            push.srcStride0 = static_cast<int32_t>(info.strideElements[0]);
            if (info.bindingCount > 2)
                push.srcStride1 = static_cast<int32_t>(info.strideElements[1]);
            if (info.bindingCount > 3)
                push.srcStride2 = static_cast<int32_t>(info.strideElements[2]);
            break;
        case kAccumulate:
            push.gate = std::abs(m.qk) <= static_cast<int>(info.frameParams[0]) ? 1u : 0u;
            break;
        case kFinish:
            push.dstStride = static_cast<int32_t>(info.dstStrideElements());
            break;
        default:
            break;
        }
        std::memcpy(pushData, &push, sizeof(push));
    };

    std::vector<VSFilterDependency> deps;
    const int rp = temporal ? rpGeneral : rpStrictSpatial;
    deps.push_back({ desc.nodes[0], rp });
    if (haveRef)
        deps.push_back({ desc.nodes[1], rp });

    std::string createError;
    VSNode *result = vsgpu::createFilter("KNLMeans", desc, deps.data(), static_cast<int>(deps.size()),
        core, vsapi, createError);
    if (result)
        vsapi->mapConsumeNode(out, "clip", result, maAppend);
    else
        vsapi->mapSetError(out, ("KNLMeans: " + createError).c_str());
}

} // namespace

VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.khanattila.knlmeansvk", "knlmvk",
        "Non-local means denoising on the VapourSynth Vulkan device",
        VS_MAKE_VERSION(KNLMVK_MAJOR_VERSION, KNLMVK_MINOR_VERSION), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("KNLMeans",
        "clip:vnode:gpu;d:int:opt;a:int:opt;s:int:opt;h:float:opt;channels:data:opt;"
        "wmode:int:opt;wref:float:opt;rclip:vnode:gpu:opt;",
        "clip:vnode:gpu;", KNLMeansCreate, nullptr, plugin);
}
