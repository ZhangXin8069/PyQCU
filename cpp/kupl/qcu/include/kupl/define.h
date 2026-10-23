#ifndef KUPL_QCU_DEFINE_H
#define KUPL_QCU_DEFINE_H

#include <cstdint>

namespace kupl::qcu {

inline constexpr int kParamsSize = 58;
inline constexpr int kArgvSize = 8;
inline constexpr int kSetPtrsSize = 100;

// Flat ABI indices shared with PyQCU/QCU.
inline constexpr int kLatX = 0;
inline constexpr int kLatY = 1;
inline constexpr int kLatZ = 2;
inline constexpr int kLatT = 3;
inline constexpr int kLatXyzt = 4;
inline constexpr int kGridX = 5;
inline constexpr int kGridY = 6;
inline constexpr int kGridZ = 7;
inline constexpr int kGridT = 8;
inline constexpr int kParity = 9;
inline constexpr int kNodeRank = 10;
inline constexpr int kNodeSize = 11;
inline constexpr int kDagger = 12;
inline constexpr int kMaxIter = 13;
inline constexpr int kDataType = 14;
inline constexpr int kLatC64 = 2;
inline constexpr int kLatC128 = 3;
inline constexpr int kSetIndex = 15;
inline constexpr int kSetPlan = 16;
inline constexpr int kSetPlanLaplacian = -2;
inline constexpr int kSetPlanGauge = -1;
inline constexpr int kSetPlanWilson = 0;
inline constexpr int kSetPlanSolver = 1;
inline constexpr int kSetPlanClover = 2;
inline constexpr int kVerbose = 51;
inline constexpr int kSeed = 52;
inline constexpr int kUseInitialGuess = 57;
inline constexpr int kMgNumLevel = 17;
inline constexpr int kMgLevel1E = 19;
inline constexpr int kMgLevel1X = 20;
inline constexpr int kMgLevel1Y = 21;
inline constexpr int kMgLevel1Z = 22;
inline constexpr int kMgLevel1T = 23;
inline constexpr int kMgLevel1MaxIter = 24;
inline constexpr int kMgLevel1DataType = 25;
inline constexpr int kMgLevel1NumRestart = 26;
inline constexpr int kMgParamsSize = 8;
inline constexpr int kMgUseGcr = 54;
inline constexpr int kMgUseDeflate = 55;
inline constexpr int kMgMuPre = 56;
inline constexpr int kMgModeGcr = 1;
inline constexpr int kMgModeMrSmoother = 2;
inline constexpr int kMgModeChebyshev = 4;
inline constexpr int kMgModeCaGcr = 8;
inline constexpr int kMgModeWCycle = 16;
inline constexpr int kMgModeFCycle = 32;
inline constexpr int kMgModeKCycle = 64;
inline constexpr int kMgModeBicgstabL = 128;
inline constexpr int kSetPtrsCoarseBase = 30;
inline constexpr int kSetPtrsStrictCoarseBase = 60;
inline constexpr int kSetPtrsStrictStride = 4;
inline constexpr int kSetPtrsStrictNull = 0;
inline constexpr int kSetPtrsStrictRawLinks = 1;
inline constexpr int kSetPtrsStrictPreconditionedLinks = 2;
inline constexpr int kSetPtrsStrictOnsitePair = 3;
inline constexpr int kSetPtrsStrictHierarchy = 80;
inline constexpr int kSetPtrsLegacyLayout = 99;
inline constexpr long long kLegacyLayoutFullSite = 0;
inline constexpr long long kLegacyLayoutOddSchur = 1;

inline constexpr int kMass = 0;
inline constexpr int kAtol = 1;
inline constexpr int kSigma = 2;
inline constexpr int kMgLevel1Atol = 3;
inline constexpr int kMgLevel2Atol = 4;
inline constexpr int kMgLevel3Atol = 5;
inline constexpr int kMgLevel4Atol = 6;
inline constexpr int kCsw = 7;
inline constexpr int kVectorSize = 12;

static_assert(sizeof(int32_t) == 4, "QCU params require 32-bit integers");
static_assert(kParamsSize == 58, "QCU params ABI must remain 58 entries");
static_assert(kArgvSize == 8, "QCU argv ABI must remain 8 entries");
static_assert(kSetPtrsSize == 100, "QCU set_ptrs ABI must remain 100 entries");
static_assert(kVectorSize == 12, "Wilson spin-color vectors have 12 entries");

} // namespace kupl::qcu

#endif
