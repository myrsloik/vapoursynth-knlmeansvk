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
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
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
enum Kernel { kZero, kPack, kWeight, kAccumulate, kFinish, kNumKernels };

/* The search runs in rounds of up to this many offsets per dispatch, dividing the
   dispatch and barrier count by the batch size. Eight triple-int offsets is what fits the
   128-byte push constant floor beside the round header. KNLMVK_BATCH=1..8 overrides the
   batch size per instance, mostly for measuring. */
constexpr int maxBatch = 8;

/* Must mirror the two push_constant blocks in shader.comp field for field. */
struct SearchPush {
    int32_t width, height;
    int32_t scrStride;
    int32_t numPairs;
    uint32_t gateMask;
    int32_t qx[maxBatch], qy[maxBatch], qk[maxBatch];
};
static_assert(sizeof(SearchPush) <= 128, "push constants have a 128 byte floor");

struct BasePush {
    int32_t width, height;
    int32_t scrStride;
    int32_t dstStride;
    int32_t srcStride0, srcStride1, srcStride2;
    int32_t layer;
};

/* One entry per pass, consulted by fillPush. */
struct PMeta {
    Kernel kind = kZero;
    int layer = 0;                         /* pack: stack layer; finish: channel */
    std::vector<std::array<int, 3>> pairs; /* search rounds: their (qx, qy, qk) */
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
        "#define KNLM_MAX_BATCH " + std::to_string(maxBatch) + "\n" +
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

    /* The half-window offsets, in the OpenCL enqueue order, then cut into rounds. */
    std::vector<std::array<int, 3>> offsets;
    {
        const int side = 2 * aTmp + 1, area = side * side;
        for (int k = -dTmp; k <= 0; k++)
            for (int j = -aTmp; j <= aTmp; j++)
                for (int i = -aTmp; i <= aTmp; i++)
                    if (k * area + j * side + i < 0)
                        offsets.push_back({ i, j, k });
    }

    /* Scratch rows use their own stride: 64 floats keeps rows 256-byte aligned and
       independent of whatever stride the output plane's sample size produces. */
    const int scrStride = (procW + 63) & ~63;
    const VkDeviceSize planeBytes = static_cast<VkDeviceSize>(scrStride) * procH * sizeof(float);

    /* Round size. Before the weight kernel was fused this tracked cache residency, since
       the box passes re-read the weight plane once per tap; now it is written once and
       read about twice, and the measured optimum stopped following any rule worth
       encoding -- on the test card the best batch is 4 at 720p and 1080p, 2 at 1440p and
       8 at 2160p, which is not even monotonic. A flat 4 is the best or within a few
       percent of it at every one of those, and a residency rule fitted to the old
       behaviour was 40% off at 2160p. KNLMVK_BATCH=1..8 overrides per machine. */
    int batch = 4;
    if (const char *forced = std::getenv("KNLMVK_BATCH"))
        batch = std::clamp(std::atoi(forced), 1, maxBatch);
    const int pairsMax = static_cast<int>(std::min<size_t>(batch, offsets.size()));

    /* Output rows per thread in the weight kernel: the more a tile covers, the less of
       its halo is recomputed border, but the more shared memory a workgroup holds and so
       the fewer of them stay resident. Four is where the two meet -- measured at 1080p,
       234 fps against 182 at one row and 178 at eight, and eight is what a
       "largest that fits shared memory" rule would have chosen. Smaller values exist only
       as a fallback for a device too small for four, which needs 15 KiB at the largest
       patch and so is within the 16 KiB Vulkan guarantees. KNLMVK_ROWS overrides. */
    uint32_t sharedLimit = 16384;
    {
        char verr[512] = {};
        VSVulkanCoreHandles vh;
        const VSVULKANAPI *vkapi = vsapi->getVulkanAPI();
        const VSVulkanFunctions *vk = vkapi ? vkapi->getVulkanFunctions(core, verr, sizeof(verr)) : nullptr;
        if (vk && !vkapi->getVulkanHandles(core, &vh, verr, sizeof(verr))) {
            VkPhysicalDeviceProperties2 props = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
            vk->vkGetPhysicalDeviceProperties2(vh.physicalDevice, &props);
            sharedLimit = props.properties.limits.maxComputeSharedMemorySize;
        }
        /* An unusable GPU API is reported with a proper message by createFilter. */
    }
    auto sharedBytesFor = [&](int rows) {
        const uint32_t padW = 16 + 2 * static_cast<uint32_t>(sTmp);
        const uint32_t padH = 16 * static_cast<uint32_t>(rows) + 2 * static_cast<uint32_t>(sTmp);
        return static_cast<uint32_t>((padH * padW + padH * 16) * sizeof(float));
    };
    int rows = 1;
    for (int r : { 4, 2, 1 }) {
        if (sharedBytesFor(r) <= sharedLimit) {
            rows = r;
            break;
        }
    }
    if (const char *forced = std::getenv("KNLMVK_ROWS"))
        rows = std::clamp(std::atoi(forced), 1, 8);
    if (sharedBytesFor(rows) > sharedLimit)
        return fail("the patch radius needs more shared memory than this device has");

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
    const int slotU2 = slotU4a + 1;
    const int slotU5 = slotU2 + 1;
    desc.scratchCount = slotU5 + 1;
    desc.scratchDefs.resize(static_cast<size_t>(desc.scratchCount));
    desc.scratchDefs[slotU1a] = { static_cast<VkDeviceSize>(T) * C * planeBytes, 0 };
    if (haveRef)
        desc.scratchDefs[slotU1b] = { static_cast<VkDeviceSize>(T) * C * planeBytes, 0 };
    desc.scratchDefs[slotU4a] = { static_cast<VkDeviceSize>(pairsMax) * 2 * planeBytes, 0 };
    /* U2 is interleaved and widened to a power of two; see U2VEC in the kernel. */
    desc.scratchDefs[slotU2] = { static_cast<VkDeviceSize>(C == 1 ? 2 : 4) * planeBytes, 0 };
    desc.scratchDefs[slotU5] = { planeBytes, 0 };

    /* Programs: the same source text specialized per kernel; parameters that are values
       rather than structure ride in as specialization constants, so instances differing
       only in parameters share every cached text. */
    const double sSize = static_cast<double>(2 * sTmp + 1) * (2 * sTmp + 1);
    const float h2InvNorm = static_cast<float>(255.0 * 255.0 / (3.0 * h * h * sSize));
    const float maxVal = fmt.sampleType == stInteger
        ? static_cast<float>((1 << fmt.bitsPerSample) - 1) : 1.0f;
    const int storageCounts[kNumKernels] = { 2, C + 1, 2, 4, 4 };
    for (int kern = 0; kern < kNumKernels; kern++) {
        const bool search = kern == kWeight || kern == kAccumulate;
        vsgpu::Program prog;
        prog.glsl = composeKernel(kern, C, fmt);
        prog.storageBufferCount = storageCounts[kern];
        prog.pushConstantBytes = search ? sizeof(SearchPush) : sizeof(BasePush);
        prog.localSizeX = 16;
        prog.localSizeY = 16;
        struct SpecData {
            int32_t refMode, wmode, patch, radius;
            float h2, wref, denorm;
            int32_t rows;
        } spec = { refMode, wmode, sTmp, dTmp, h2InvNorm, static_cast<float>(wref), maxVal, rows };
        prog.specData.assign(reinterpret_cast<const uint8_t *>(&spec),
            reinterpret_cast<const uint8_t *>(&spec) + sizeof(spec));
        for (uint32_t i = 0; i < 8; i++)
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

    /* The search, in rounds of up to `batch` offsets: one weight pass and one accumulate
       pass cover a whole round. The offsets keep the OpenCL enqueue order; within a round
       the accumulate sums its pairs in registers before touching U2, which only regroups
       the float additions. */
    const int distStack = haveRef ? slotU1b : slotU1a;
    for (size_t o = 0; o < offsets.size(); o += static_cast<size_t>(batch)) {
        const size_t oEnd = std::min(o + static_cast<size_t>(batch), offsets.size());
        std::vector<std::array<int, 3>> round(offsets.begin() + o, offsets.begin() + oEnd);
        const std::pair<Kernel, std::vector<int>> roundPasses[2] = {
            { kWeight, { distStack, slotU4a } },
            { kAccumulate, { slotU1a, slotU4a, slotU2, slotU5 } },
        };
        for (const auto &[kern, slots] : roundPasses) {
            vsgpu::Pass pass;
            pass.program = kern;
            for (int slot : slots)
                pass.bindings.push_back(vsgpu::Operand::scratch(slot));
            gateHeavy(pass);
            /* The weight kernel emits ROWS rows per thread, so its grid is that much
               shorter; it bounds its stores against the real dimensions, which fillPush
               passes from creation-time constants rather than from the reshaped grid. */
            if (kern == kWeight && rows > 1)
                pass.reshape = [rows](vsgpu::PassInfo &info) {
                    info.height = (info.height + rows - 1) / rows;
                };
            desc.passes.push_back(std::move(pass));
            PMeta m;
            m.kind = kern;
            m.pairs = round;
            meta.push_back(std::move(m));
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
    desc.fillPush = [meta, scrStrideV, procW, procH](const vsgpu::PassInfo &info, void *pushData) {
        const PMeta &m = meta[info.pass];
        if (m.kind == kWeight || m.kind == kAccumulate) {
            SearchPush push = {};
            /* Creation-time dimensions, not info's: the weight pass reshapes its grid. */
            push.width = procW;
            push.height = procH;
            push.scrStride = scrStrideV;
            push.numPairs = static_cast<int32_t>(m.pairs.size());
            const int kv = static_cast<int>(info.frameParams[0]);
            for (size_t p = 0; p < m.pairs.size(); p++) {
                push.qx[p] = m.pairs[p][0];
                push.qy[p] = m.pairs[p][1];
                push.qk[p] = m.pairs[p][2];
                if (std::abs(m.pairs[p][2]) <= kv)
                    push.gateMask |= 1u << p;
            }
            std::memcpy(pushData, &push, sizeof(push));
            return;
        }
        BasePush push = {};
        push.width = static_cast<int32_t>(info.width);
        push.height = static_cast<int32_t>(info.height);
        push.scrStride = scrStrideV;
        push.layer = m.layer;
        if (m.kind == kPack) {
            push.srcStride0 = static_cast<int32_t>(info.strideElements[0]);
            if (info.bindingCount > 2)
                push.srcStride1 = static_cast<int32_t>(info.strideElements[1]);
            if (info.bindingCount > 3)
                push.srcStride2 = static_cast<int32_t>(info.strideElements[2]);
        } else if (m.kind == kFinish) {
            push.dstStride = static_cast<int32_t>(info.dstStrideElements());
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
