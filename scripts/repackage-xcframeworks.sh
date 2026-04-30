#!/usr/bin/env bash
#
# Repackage the fat LiteRTLM.xcframework into App-Store-compliant
# xcframeworks:
#
#   1. LiteRTLM.xcframework                     — CLiteRTLM only (no loose dylibs)
#   2. GemmaModelConstraintProvider.xcframework — sibling framework for the Gemma
#                                                 constraint provider dylib
#   3. LiteRtMetalAccelerator.xcframework       — sibling framework for the Metal
#                                                 accelerator dylib (dlopen'd at
#                                                 runtime by the engine)
#   4. LiteRtTopKMetalSampler.xcframework       — sibling framework for the Metal
#                                                 TopK sampler dylib (dlopen'd at
#                                                 runtime by the engine)
#   5. LiteRt.xcframework                       — libLiteRt.dylib providing _LiteRt*
#                                                 symbols in the flat namespace so
#                                                 the Metal accelerator can resolve
#                                                 them at dlopen time
#
# The fix addresses App Store Connect errors:
#   - ITMS-90171: loose .dylib files inside .framework are rejected
#   - ITMS-90057: CFBundleShortVersionString missing from CLiteRTLM Info.plist
#
# Runtime details:
#   - CLiteRTLM hard-links GemmaModelConstraintProvider via LC_LOAD_DYLIB.
#     We rewrite it to @rpath/GemmaModelConstraintProvider.framework/<binary>
#     and add LC_RPATH @loader_path/.. so dyld finds it at app's Frameworks/.
#   - The Metal plugin dylibs are dlopen'd by leaf name. LiteRTLMSwift
#     preemptively dlopens the frameworks by full path; consumers then patch
#     install_names after embed so dyld's install-name cache answers the
#     engine's leaf-name dlopen.
#   - LiteRtMetalAccelerator uses flat namespace to resolve _LiteRt* symbols.
#     LiteRt.framework exports them; it must be embedded in the app.
#
# Input:  Frameworks/LiteRTLM.xcframework (as produced by build-xcframework.sh
#         or its slice-builder mode, see --slices flag).
#         Optional env: LITERT_DEVICE_DYLIB, LITERT_SIM_DYLIB (paths to prebuilt
#         libLiteRt.dylib slices; if unset, look in already-split xcframework).
# Output: Frameworks/{LiteRTLM,GemmaModelConstraintProvider,LiteRtMetalAccelerator,LiteRtTopKMetalSampler,LiteRt}.xcframework

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FRAMEWORKS_DIR="$PROJECT_DIR/Frameworks"
SRC_XCF="$FRAMEWORKS_DIR/LiteRTLM.xcframework"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# Rewrite LC_BUILD_VERSION so the dylib advertises iOS 17 as its minimum.
# Google's prebuilt Gemma dylib ships with minos=26.2 which would refuse to
# load on any device running iOS < 26.2 (the minimum mei supports is 17.0).
# Metal dylib already has minos=14.0 so this is a no-op for it.
# -tool 3 1230.1 adds an ld tool entry; App Store validation (ITMS-90208)
# rejects framework binaries whose LC_BUILD_VERSION has ntools=0.
set_ios_min() {
    local BIN="$1" PLATFORM="$2"  # PLATFORM: ios or iossim
    xcrun vtool -set-build-version "$PLATFORM" 17.0 26.2 -tool 3 1230.1 -replace -output "$BIN" "$BIN" >/dev/null
}

[ -d "$SRC_XCF" ] || error "Source xcframework missing: $SRC_XCF"
[ -d "$SRC_XCF/ios-arm64/CLiteRTLM.framework" ] || error "Missing device slice"
[ -d "$SRC_XCF/ios-arm64-simulator/CLiteRTLM.framework" ] || error "Missing sim slice"

DEVICE_SRC="$SRC_XCF/ios-arm64/CLiteRTLM.framework"
SIM_SRC="$SRC_XCF/ios-arm64-simulator/CLiteRTLM.framework"

# Source of the plugin dylibs. build-xcframework.sh produces a fat layout
# where they live loosely inside CLiteRTLM.framework; re-runs of this script
# find them in the already-split xcframeworks instead.
GEMMA_DEVICE_DYLIB="$DEVICE_SRC/libGemmaModelConstraintProvider.dylib"
if [ ! -f "$GEMMA_DEVICE_DYLIB" ]; then
    GEMMA_DEVICE_DYLIB="$FRAMEWORKS_DIR/GemmaModelConstraintProvider.xcframework/ios-arm64/GemmaModelConstraintProvider.framework/GemmaModelConstraintProvider"
fi
find_existing_metal() {
    # Look in both the current (renamed) location and the previous location
    # so the script stays idempotent across structure changes.
    local SLICE="$1"
    for CAND in \
        "$FRAMEWORKS_DIR/LiteRtMetalAccelerator.xcframework/$SLICE/LiteRtMetalAccelerator.framework/libLiteRtMetalAccelerator.dylib" \
        "$FRAMEWORKS_DIR/LiteRtMetalAccelerator.xcframework/$SLICE/LiteRtMetalAccelerator.framework/LiteRtMetalAccelerator"
    do
        [ -f "$CAND" ] && { echo "$CAND"; return; }
    done
}
find_existing_topk() {
    local SLICE="$1"
    for CAND in \
        "$FRAMEWORKS_DIR/LiteRtTopKMetalSampler.xcframework/$SLICE/LiteRtTopKMetalSampler.framework/libLiteRtTopKMetalSampler.dylib" \
        "$FRAMEWORKS_DIR/LiteRtTopKMetalSampler.xcframework/$SLICE/LiteRtTopKMetalSampler.framework/LiteRtTopKMetalSampler"
    do
        [ -f "$CAND" ] && { echo "$CAND"; return; }
    done
}
find_existing_litert() {
    local SLICE="$1"
    for CAND in \
        "$FRAMEWORKS_DIR/LiteRt.xcframework/$SLICE/LiteRt.framework/LiteRt"
    do
        [ -f "$CAND" ] && { echo "$CAND"; return; }
    done
}
METAL_DEVICE_DYLIB="$DEVICE_SRC/libLiteRtMetalAccelerator.dylib"
[ -f "$METAL_DEVICE_DYLIB" ] || METAL_DEVICE_DYLIB="$(find_existing_metal ios-arm64)"
METAL_SIM_DYLIB="$SIM_SRC/libLiteRtMetalAccelerator.dylib"
[ -f "$METAL_SIM_DYLIB" ] || METAL_SIM_DYLIB="$(find_existing_metal ios-arm64-simulator)"
TOPK_DEVICE_DYLIB="$DEVICE_SRC/libLiteRtTopKMetalSampler.dylib"
[ -f "$TOPK_DEVICE_DYLIB" ] || TOPK_DEVICE_DYLIB="$(find_existing_topk ios-arm64)"
# LiteRt prebuilt: passed from build-xcframework.sh via env, or found in already-split xcframework
LITERT_DEVICE_DYLIB="${LITERT_DEVICE_DYLIB:-}"
LITERT_SIM_DYLIB="${LITERT_SIM_DYLIB:-}"
[ -f "$LITERT_DEVICE_DYLIB" ] || LITERT_DEVICE_DYLIB="$(find_existing_litert ios-arm64)"
[ -f "$LITERT_SIM_DYLIB" ]    || LITERT_SIM_DYLIB="$(find_existing_litert ios-arm64-simulator)"
[ -f "$GEMMA_DEVICE_DYLIB" ] || error "Gemma device dylib not found"
[ -f "$METAL_DEVICE_DYLIB" ] || error "Metal device dylib not found"
[ -f "$METAL_SIM_DYLIB" ]    || error "Metal sim dylib not found"
[ -f "$TOPK_DEVICE_DYLIB" ]  || error "TopK Metal sampler device dylib not found"
[ -f "$LITERT_DEVICE_DYLIB" ] || error "LiteRt device dylib not found"
[ -f "$LITERT_SIM_DYLIB" ]    || error "LiteRt sim dylib not found"

# ---------------------------------------------------------------------------
# 1. Clean CLiteRTLM.framework slices
# ---------------------------------------------------------------------------
#   - Strip loose plugin dylibs (ITMS-90171)
#   - Add CFBundleShortVersionString (ITMS-90057)
#   - Device slice only: rewrite Gemma LC_LOAD_DYLIB, add sibling-Frameworks
#     rpath

write_plist() {
    local DIR="$1" NAME="$2" IDENT="$3"
    local EXEC="${4:-$NAME}"
    cat > "$DIR/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleExecutable</key><string>$EXEC</string>
    <key>CFBundleIdentifier</key><string>$IDENT</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleName</key><string>$NAME</string>
    <key>CFBundlePackageType</key><string>FMWK</string>
    <key>CFBundleShortVersionString</key><string>1.0.0</string>
    <key>CFBundleSupportedPlatforms</key>
    <array><string>iPhoneOS</string></array>
    <key>CFBundleVersion</key><string>1</string>
    <key>DTCompiler</key><string>com.apple.compilers.llvm.clang.1_0</string>
    <key>DTPlatformName</key><string>iphoneos</string>
    <key>DTPlatformVersion</key><string>17.0</string>
    <key>DTSDKName</key><string>iphoneos17.0</string>
    <key>MinimumOSVersion</key><string>17.0</string>
</dict>
</plist>
PLIST
}

clean_core_slice() {
    local SLICE="$1" SRC="$2" IS_DEVICE="$3"
    local DEST="$WORK_DIR/core/$SLICE/CLiteRTLM.framework"
    mkdir -p "$(dirname "$DEST")"
    cp -R "$SRC" "$DEST"
    rm -f "$DEST"/libLiteRtMetalAccelerator.dylib \
          "$DEST"/libLiteRtTopKMetalSampler.dylib \
          "$DEST"/libGemmaModelConstraintProvider.dylib
    rm -rf "$DEST/_CodeSignature"
    write_plist "$DEST" "CLiteRTLM" "com.google.CLiteRTLM"
    if [ "$IS_DEVICE" = "yes" ]; then
        install_name_tool -change \
            "@rpath/libGemmaModelConstraintProvider.dylib" \
            "@rpath/GemmaModelConstraintProvider.framework/GemmaModelConstraintProvider" \
            "$DEST/CLiteRTLM"
        # Add @loader_path/.. so sibling frameworks resolve from app Frameworks/.
        install_name_tool -add_rpath "@loader_path/.." "$DEST/CLiteRTLM" 2>/dev/null || true
    fi
    codesign --force --sign - "$DEST/CLiteRTLM"
    info "Cleaned CLiteRTLM slice: $SLICE"
}

clean_core_slice "ios-arm64" "$DEVICE_SRC" "yes"
clean_core_slice "ios-arm64-simulator" "$SIM_SRC" "no"

# ---------------------------------------------------------------------------
# 2. Build GemmaModelConstraintProvider.framework
# ---------------------------------------------------------------------------
#   Device: real dylib, install_name rewritten to @rpath/.framework/<binary>
#   Sim:    empty clang stub (CLiteRTLM sim slice has no LC_LOAD_DYLIB for
#           Gemma, so the stub is never actually linked — it exists only so
#           SPM can resolve the xcframework for sim builds).

make_gemma_slice() {
    local SLICE="$1" SRC_DYLIB="$2"
    local DEST="$WORK_DIR/gemma/$SLICE/GemmaModelConstraintProvider.framework"
    mkdir -p "$DEST"
    cp "$SRC_DYLIB" "$DEST/GemmaModelConstraintProvider"
    install_name_tool -id \
        "@rpath/GemmaModelConstraintProvider.framework/GemmaModelConstraintProvider" \
        "$DEST/GemmaModelConstraintProvider"
    case "$SLICE" in
        ios-arm64)           set_ios_min "$DEST/GemmaModelConstraintProvider" ios ;;
        ios-arm64-simulator) set_ios_min "$DEST/GemmaModelConstraintProvider" iossim ;;
    esac
    write_plist "$DEST" "GemmaModelConstraintProvider" "com.google.GemmaModelConstraintProvider"
    codesign --force --sign - "$DEST/GemmaModelConstraintProvider"
    info "Built Gemma slice: $SLICE"
}

make_gemma_slice "ios-arm64" "$GEMMA_DEVICE_DYLIB"

GEMMA_STUB_SRC="$WORK_DIR/gemma_stub.c"
GEMMA_STUB_DYLIB="$WORK_DIR/gemma_stub.dylib"
cat > "$GEMMA_STUB_SRC" << 'EOF'
/* Simulator-only stub. CLiteRTLM's sim slice does not link this dylib;
   it exists so SPM has a simulator slice to resolve. */
int __gemma_model_constraint_provider_stub(void) { return 0; }
EOF
xcrun --sdk iphonesimulator clang \
    -target arm64-apple-ios17-simulator \
    -dynamiclib \
    -install_name "@rpath/GemmaModelConstraintProvider.framework/GemmaModelConstraintProvider" \
    "$GEMMA_STUB_SRC" -o "$GEMMA_STUB_DYLIB"
make_gemma_slice "ios-arm64-simulator" "$GEMMA_STUB_DYLIB"

# ---------------------------------------------------------------------------
# 3. Build LiteRtMetalAccelerator.framework
# ---------------------------------------------------------------------------
#   Both slices from the original xcframework's copies.
#   Install_name is set to the BARE LEAF "libLiteRtMetalAccelerator.dylib"
#   so dyld's install-name cache (populated by LiteRTLMSwift's preemptive
#   dlopen by full path) satisfies the engine's leaf-name dlopen.

make_metal_slice() {
    local SLICE="$1" SRC_DYLIB="$2"
    local DEST="$WORK_DIR/metal/$SLICE/LiteRtMetalAccelerator.framework"
    mkdir -p "$DEST"
    # Standard framework layout so `ld -framework LiteRtMetalAccelerator` works.
    # The engine dlopens by the bare-leaf "libLiteRtMetalAccelerator.dylib"
    # at runtime; consumers patch the embedded binary's install_name to that
    # leaf after embed (see repo README / mei build phase) so dyld's cache
    # lookup cache-hits on the leaf after the framework is loaded at launch.
    cp "$SRC_DYLIB" "$DEST/LiteRtMetalAccelerator"
    install_name_tool -id \
        "@rpath/LiteRtMetalAccelerator.framework/LiteRtMetalAccelerator" \
        "$DEST/LiteRtMetalAccelerator"
    case "$SLICE" in
        ios-arm64)           set_ios_min "$DEST/LiteRtMetalAccelerator" ios ;;
        ios-arm64-simulator) set_ios_min "$DEST/LiteRtMetalAccelerator" iossim ;;
    esac
    write_plist "$DEST" "LiteRtMetalAccelerator" "com.google.LiteRtMetalAccelerator"
    codesign --force --sign - "$DEST/LiteRtMetalAccelerator"
    info "Built Metal slice: $SLICE"
}

make_metal_slice "ios-arm64" "$METAL_DEVICE_DYLIB"
make_metal_slice "ios-arm64-simulator" "$METAL_SIM_DYLIB"

# ---------------------------------------------------------------------------
# 4. Build LiteRtTopKMetalSampler.framework
# ---------------------------------------------------------------------------
#   Device: real upstream prebuilt from LiteRT-LM prebuilt/ios_arm64.
#   Sim:    empty clang stub. Upstream does not currently ship an iOS simulator
#           TopK Metal sampler, and the stub exists only so SPM can resolve the
#           binary target for simulator builds.

make_topk_slice() {
    local SLICE="$1" SRC_DYLIB="$2"
    local DEST="$WORK_DIR/topk/$SLICE/LiteRtTopKMetalSampler.framework"
    mkdir -p "$DEST"
    cp "$SRC_DYLIB" "$DEST/LiteRtTopKMetalSampler"
    # The engine calls dlopen("libLiteRtTopKMetalSampler.dylib") by bare leaf name.
    # Setting the install name to that leaf means dyld registers the library under
    # this name at launch (loaded via LC_LOAD_DYLIB from the app binary), so the
    # engine's bare-name dlopen finds it already loaded.
    install_name_tool -id \
        "libLiteRtTopKMetalSampler.dylib" \
        "$DEST/LiteRtTopKMetalSampler"
    case "$SLICE" in
        ios-arm64)           set_ios_min "$DEST/LiteRtTopKMetalSampler" ios ;;
        ios-arm64-simulator) set_ios_min "$DEST/LiteRtTopKMetalSampler" iossim ;;
    esac
    write_plist "$DEST" "LiteRtTopKMetalSampler" "com.google.LiteRtTopKMetalSampler"
    codesign --force --sign - "$DEST/LiteRtTopKMetalSampler"
    info "Built TopK Metal sampler slice: $SLICE"
}

make_topk_slice "ios-arm64" "$TOPK_DEVICE_DYLIB"

TOPK_STUB_SRC="$WORK_DIR/topk_stub.c"
TOPK_STUB_DYLIB="$WORK_DIR/topk_stub.dylib"
cat > "$TOPK_STUB_SRC" << 'EOF'
/* Simulator-only stub. Upstream currently ships only an iOS device
   libLiteRtTopKMetalSampler.dylib. */
void* LiteRtTopKMetalSampler_Create(void) { return 0; }
void LiteRtTopKMetalSampler_Destroy(void* sampler) { (void)sampler; }
int LiteRtTopKMetalSampler_SampleToIdAndScoreBuffer(void) { return -1; }
EOF
xcrun --sdk iphonesimulator clang \
    -target arm64-apple-ios17-simulator \
    -dynamiclib \
    -install_name "@rpath/LiteRtTopKMetalSampler.framework/LiteRtTopKMetalSampler" \
    "$TOPK_STUB_SRC" -o "$TOPK_STUB_DYLIB"
make_topk_slice "ios-arm64-simulator" "$TOPK_STUB_DYLIB"

# ---------------------------------------------------------------------------
# 5. Build LiteRt.framework
# ---------------------------------------------------------------------------
#   Both slices from prebuilt libLiteRt.dylib (passed via LITERT_DEVICE_DYLIB /
#   LITERT_SIM_DYLIB env vars set by build-xcframework.sh, or found in the
#   already-split xcframework on re-runs).
#
#   LiteRtMetalAccelerator uses flat namespace to look up _LiteRt* symbols.
#   This framework exports them so they're available in the process flat namespace
#   at the point the Metal accelerator is dlopen'd.

make_litert_slice() {
    local SLICE="$1" SRC_DYLIB="$2"
    local DEST="$WORK_DIR/litert/$SLICE/LiteRt.framework"
    mkdir -p "$DEST"
    cp "$SRC_DYLIB" "$DEST/LiteRt"
    install_name_tool -id \
        "@rpath/LiteRt.framework/LiteRt" \
        "$DEST/LiteRt"
    case "$SLICE" in
        ios-arm64)           set_ios_min "$DEST/LiteRt" ios ;;
        ios-arm64-simulator) set_ios_min "$DEST/LiteRt" iossim ;;
    esac
    write_plist "$DEST" "LiteRt" "com.google.LiteRt"
    codesign --force --sign - "$DEST/LiteRt"
    info "Built LiteRt slice: $SLICE"
}

make_litert_slice "ios-arm64" "$LITERT_DEVICE_DYLIB"
make_litert_slice "ios-arm64-simulator" "$LITERT_SIM_DYLIB"

# ---------------------------------------------------------------------------
# 6. Create the five xcframeworks
# ---------------------------------------------------------------------------

rm -rf "$FRAMEWORKS_DIR/LiteRTLM.xcframework" \
       "$FRAMEWORKS_DIR/GemmaModelConstraintProvider.xcframework" \
       "$FRAMEWORKS_DIR/LiteRtMetalAccelerator.xcframework" \
       "$FRAMEWORKS_DIR/LiteRtTopKMetalSampler.xcframework" \
       "$FRAMEWORKS_DIR/LiteRt.xcframework"

xcodebuild -create-xcframework \
    -framework "$WORK_DIR/core/ios-arm64/CLiteRTLM.framework" \
    -framework "$WORK_DIR/core/ios-arm64-simulator/CLiteRTLM.framework" \
    -output "$FRAMEWORKS_DIR/LiteRTLM.xcframework" >/dev/null

xcodebuild -create-xcframework \
    -framework "$WORK_DIR/gemma/ios-arm64/GemmaModelConstraintProvider.framework" \
    -framework "$WORK_DIR/gemma/ios-arm64-simulator/GemmaModelConstraintProvider.framework" \
    -output "$FRAMEWORKS_DIR/GemmaModelConstraintProvider.xcframework" >/dev/null

xcodebuild -create-xcframework \
    -framework "$WORK_DIR/metal/ios-arm64/LiteRtMetalAccelerator.framework" \
    -framework "$WORK_DIR/metal/ios-arm64-simulator/LiteRtMetalAccelerator.framework" \
    -output "$FRAMEWORKS_DIR/LiteRtMetalAccelerator.xcframework" >/dev/null

xcodebuild -create-xcframework \
    -framework "$WORK_DIR/topk/ios-arm64/LiteRtTopKMetalSampler.framework" \
    -framework "$WORK_DIR/topk/ios-arm64-simulator/LiteRtTopKMetalSampler.framework" \
    -output "$FRAMEWORKS_DIR/LiteRtTopKMetalSampler.xcframework" >/dev/null

xcodebuild -create-xcframework \
    -framework "$WORK_DIR/litert/ios-arm64/LiteRt.framework" \
    -framework "$WORK_DIR/litert/ios-arm64-simulator/LiteRt.framework" \
    -output "$FRAMEWORKS_DIR/LiteRt.xcframework" >/dev/null

info "Wrote:"
info "  $FRAMEWORKS_DIR/LiteRTLM.xcframework"
info "  $FRAMEWORKS_DIR/GemmaModelConstraintProvider.xcframework"
info "  $FRAMEWORKS_DIR/LiteRtMetalAccelerator.xcframework"
info "  $FRAMEWORKS_DIR/LiteRtTopKMetalSampler.xcframework"
info "  $FRAMEWORKS_DIR/LiteRt.xcframework"
