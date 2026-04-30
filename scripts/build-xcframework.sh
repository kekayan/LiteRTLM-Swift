#!/usr/bin/env bash
#
# Build LiteRTLM Swift binary frameworks from Google's LiteRT-LM source.
#
# Prerequisites:
#   - Bazel 7.6.1 (install via Bazelisk: brew install bazelisk)
#   - Xcode 16+ with iOS SDK
#   - ~20 GB disk space for Bazel build cache
#
# Usage:
#   ./scripts/build-xcframework.sh [/path/to/LiteRT-LM]
#
# Optional environment:
#   LITERT_LM_REF=<git ref>       Checkout this upstream ref when cloning.
#
# If no path is provided, clones the repo to a temp directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/Frameworks/LiteRTLM.xcframework"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Locate or clone LiteRT-LM source
# ---------------------------------------------------------------------------

LITERT_LM_DIR="${1:-}"

if [ -z "$LITERT_LM_DIR" ]; then
    LITERT_LM_DIR="$WORK_DIR/LiteRT-LM"
    info "Cloning LiteRT-LM source..."
    if [ -n "${LITERT_LM_REF:-}" ]; then
        git clone --filter=blob:none --no-checkout https://github.com/google-ai-edge/LiteRT-LM.git "$LITERT_LM_DIR"
        git -C "$LITERT_LM_DIR" fetch --depth 1 origin "$LITERT_LM_REF"
        git -C "$LITERT_LM_DIR" checkout --detach FETCH_HEAD
    else
        git clone --depth 1 https://github.com/google-ai-edge/LiteRT-LM.git "$LITERT_LM_DIR"
    fi
fi

if [ ! -f "$LITERT_LM_DIR/c/BUILD" ]; then
    error "Invalid LiteRT-LM source directory: $LITERT_LM_DIR (missing c/BUILD)"
fi

# Resolve to absolute path (relative paths break after `cd` into the source dir)
LITERT_LM_DIR="$(cd "$LITERT_LM_DIR" && pwd)"

UPSTREAM_REV="$(git -C "$LITERT_LM_DIR" rev-parse --short=12 HEAD 2>/dev/null || true)"
info "Using LiteRT-LM source at: $LITERT_LM_DIR${UPSTREAM_REV:+ ($UPSTREAM_REV)}"

# Pull all prebuilt dylibs via Git LFS (Metal accelerator, LiteRt runtime, TopK sampler)
info "Pulling prebuilt dylibs via git lfs..."
git -C "$LITERT_LM_DIR" lfs pull --include "prebuilt/ios_arm64/*" --include "prebuilt/ios_sim_arm64/*"

TOPK_CANDIDATE="$LITERT_LM_DIR/prebuilt/ios_arm64/libLiteRtTopKMetalSampler.dylib"
TOPK_DEVICE=""
if [ -f "$TOPK_CANDIDATE" ] && file "$TOPK_CANDIDATE" | grep -q 'Mach-O'; then
    TOPK_DEVICE="$TOPK_CANDIDATE"
else
    error "TopK Metal sampler prebuilt not found at $TOPK_CANDIDATE"
fi
info "TopK Metal sampler device dylib: $(du -h "$TOPK_DEVICE" | cut -f1)"

# ---------------------------------------------------------------------------
# 1b. Patch upstream BUILD if needed
# ---------------------------------------------------------------------------
# Two patches may be needed depending on the upstream version:
#
# 1. ios_engine.bzl stub — HEAD's c/BUILD loads `:ios_engine.bzl` which isn't
#    shipped yet. Without the stub Bazel can't parse the BUILD file at all.
#
# 2. cc_binary dylib target — releases up to v0.10.2 only define cc_library
#    targets (:engine, :engine_cpu). The cc_binary that produces the shared
#    library was added later. We append it if missing.

if [ ! -f "$LITERT_LM_DIR/c/ios_engine.bzl" ] && grep -q 'ios_engine\.bzl' "$LITERT_LM_DIR/c/BUILD"; then
    info "Creating stub ios_engine.bzl (missing from upstream)..."
    cat > "$LITERT_LM_DIR/c/ios_engine.bzl" << 'STUB'
"""Stub for ios_shared_engine macro (not yet published upstream)."""

def ios_shared_engine(**kwargs):
    pass
STUB
fi

# If the target already exists but has linkstatic (from a prior script run), strip
# both the linkstatic line and the old _LiteRt* exported_symbol line so we add the
# corrected version below.
if grep -q 'linkstatic = True' "$LITERT_LM_DIR/c/BUILD"; then
    info "Removing stale linkstatic from libLiteRTLMEngine.dylib target..."
    sed -i '' '/linkstatic = True,/d' "$LITERT_LM_DIR/c/BUILD"
fi
if grep -q 'exported_symbol,_LiteRt' "$LITERT_LM_DIR/c/BUILD"; then
    info "Removing stale _LiteRt* exported_symbol from libLiteRTLMEngine.dylib target..."
    sed -i '' '/exported_symbol,_LiteRt/d' "$LITERT_LM_DIR/c/BUILD"
fi

if ! grep -q 'libLiteRTLMEngine\.dylib' "$LITERT_LM_DIR/c/BUILD"; then
    info "Adding libLiteRTLMEngine.dylib target (not present in this version)..."
    # litert_lm_logging.cc/.h were removed in upstream main; include them only
    # when present (needed for older pinned commits like 40dee845).
    LOGGING_SRCS=""
    if [ -f "$LITERT_LM_DIR/c/litert_lm_logging.cc" ]; then
        LOGGING_SRCS='        "litert_lm_logging.cc",
        "litert_lm_logging.h",'
    fi
    cat >> "$LITERT_LM_DIR/c/BUILD" << BUILD_PATCH

cc_binary(
    name = "libLiteRTLMEngine.dylib",
    srcs = [
        "engine.cc",
        "engine.h",
$LOGGING_SRCS
    ],
    linkopts = [
        "-Wl,-exported_symbol,_litert_lm_*",
    ],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = ENGINE_COMMON_DEPS + [
        "//runtime/core:engine_impl",
    ],
)
BUILD_PATCH
fi

# ---------------------------------------------------------------------------
# 2. Check prerequisites
# ---------------------------------------------------------------------------

if ! command -v bazel &>/dev/null && ! command -v bazelisk &>/dev/null; then
    error "Bazel not found. Install via: brew install bazelisk"
fi

BAZEL_CMD="bazel"
if command -v bazelisk &>/dev/null; then
    BAZEL_CMD="bazelisk"
fi

if ! xcode-select -p &>/dev/null; then
    error "Xcode command line tools not found. Run: xcode-select --install"
fi

info "Using $($BAZEL_CMD --version | head -1)"
info "Using $(xcodebuild -version | head -1)"

# ---------------------------------------------------------------------------
# 3. Build for iOS device (arm64)
# ---------------------------------------------------------------------------

info "Building for iOS device (arm64)..."
cd "$LITERT_LM_DIR"

$BAZEL_CMD build --config=ios_arm64 //c:libLiteRTLMEngine.dylib 2>&1 | tail -5

DEVICE_DYLIB_SRC="$LITERT_LM_DIR/bazel-bin/c/libLiteRTLMEngine.dylib"
if [ ! -f "$DEVICE_DYLIB_SRC" ]; then
    error "Device build failed: $DEVICE_DYLIB_SRC not found"
fi
info "Device build OK: $(du -h "$DEVICE_DYLIB_SRC" | cut -f1)"

# Copy device dylib aside before sim build overwrites bazel-bin
DEVICE_DYLIB="$WORK_DIR/libLiteRTLMEngine-device.dylib"
cp "$DEVICE_DYLIB_SRC" "$DEVICE_DYLIB"

# Also grab the GemmaModelConstraintProvider dylib if present
CONSTRAINT_DYLIB=""
if [ -f "$LITERT_LM_DIR/bazel-bin/c/libGemmaModelConstraintProvider.dylib" ]; then
    CONSTRAINT_DYLIB="$WORK_DIR/libGemmaModelConstraintProvider.dylib"
    cp "$LITERT_LM_DIR/bazel-bin/c/libGemmaModelConstraintProvider.dylib" "$CONSTRAINT_DYLIB"
    info "Found libGemmaModelConstraintProvider.dylib"
fi

# ---------------------------------------------------------------------------
# 4. Build for iOS simulator (arm64)
# ---------------------------------------------------------------------------

info "Building for iOS simulator (arm64)..."

$BAZEL_CMD build --config=ios_sim_arm64 //c:libLiteRTLMEngine.dylib 2>&1 | tail -5

SIM_DYLIB_SRC="$LITERT_LM_DIR/bazel-bin/c/libLiteRTLMEngine.dylib"
if [ ! -f "$SIM_DYLIB_SRC" ]; then
    error "Simulator build failed: $SIM_DYLIB_SRC not found"
fi
info "Simulator build OK: $(du -h "$SIM_DYLIB_SRC" | cut -f1)"

SIM_DYLIB="$WORK_DIR/libLiteRTLMEngine-sim.dylib"
cp "$SIM_DYLIB_SRC" "$SIM_DYLIB"

# ---------------------------------------------------------------------------
# 4b. Locate prebuilt dylibs from the LiteRT-LM repo
# ---------------------------------------------------------------------------
# All GPU/Metal plugin dylibs ship as Git LFS prebuilts under prebuilt/ in the
# upstream repo. We pulled them above; just verify and set paths here.

METAL_DEVICE="$LITERT_LM_DIR/prebuilt/ios_arm64/libLiteRtMetalAccelerator.dylib"
METAL_SIM="$LITERT_LM_DIR/prebuilt/ios_sim_arm64/libLiteRtMetalAccelerator.dylib"

for DYLIB in "$METAL_DEVICE" "$METAL_SIM"; do
    if [ ! -f "$DYLIB" ] || ! file "$DYLIB" | grep -q 'Mach-O'; then
        error "Prebuilt dylib missing or not a Mach-O binary: $DYLIB"
    fi
done

info "Metal device dylib:  $(du -h "$METAL_DEVICE" | cut -f1)"
info "Metal sim dylib:     $(du -h "$METAL_SIM" | cut -f1)"

# ---------------------------------------------------------------------------
# 4c. LiteRT TopK Metal sampler prebuilt (device only; sim stub added later)
# ---------------------------------------------------------------------------

info "Using TopK Metal sampler from: $TOPK_DEVICE"

# ---------------------------------------------------------------------------
# 5. Package as .framework bundles
# ---------------------------------------------------------------------------

HEADERS_DIR="$LITERT_LM_DIR/c"
BUNDLE_ID="com.google.CLiteRTLM"
FRAMEWORK_NAME="CLiteRTLM"
MIN_IOS="13.0"

package_framework() {
    local ARCH_NAME="$1"  # e.g. "ios-arm64"
    local DYLIB_PATH="$2"
    local FW_DIR="$WORK_DIR/$ARCH_NAME/$FRAMEWORK_NAME.framework"
    shift 2
    local EXTRA_DYLIBS=("$@")

    mkdir -p "$FW_DIR/Headers" "$FW_DIR/Modules"

    # Copy binary (rename to framework name)
    cp "$DYLIB_PATH" "$FW_DIR/$FRAMEWORK_NAME"

    # Fix install name
    install_name_tool -id "@rpath/$FRAMEWORK_NAME.framework/$FRAMEWORK_NAME" "$FW_DIR/$FRAMEWORK_NAME"

    # Copy, re-id, and re-sign each extra plugin dylib so dyld resolves them
    # via the framework's @rpath at runtime.
    for EXTRA in "${EXTRA_DYLIBS[@]}"; do
        [ -z "$EXTRA" ] && continue
        [ -f "$EXTRA" ] || continue
        local BASENAME
        BASENAME="$(basename "$EXTRA")"
        cp "$EXTRA" "$FW_DIR/$BASENAME"
        install_name_tool -id "@rpath/$FRAMEWORK_NAME.framework/$BASENAME" "$FW_DIR/$BASENAME" || true
    done

    # Copy headers (litert_lm_logging.h absent in upstream main)
    cp "$HEADERS_DIR/engine.h" "$FW_DIR/Headers/"
    [ -f "$HEADERS_DIR/litert_lm_logging.h" ] && cp "$HEADERS_DIR/litert_lm_logging.h" "$FW_DIR/Headers/"

    # Create module map
    cat > "$FW_DIR/Modules/module.modulemap" << 'MODULEMAP'
framework module CLiteRTLM {
    header "engine.h"
    export *
}
MODULEMAP

    # Create Info.plist
    cat > "$FW_DIR/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$FRAMEWORK_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$FRAMEWORK_NAME</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>MinimumOSVersion</key>
    <string>$MIN_IOS</string>
</dict>
</plist>
PLIST

    # Ad-hoc code sign main binary + every embedded plugin dylib
    codesign --force --sign - "$FW_DIR/$FRAMEWORK_NAME"
    for EXTRA in "${EXTRA_DYLIBS[@]}"; do
        [ -z "$EXTRA" ] && continue
        local BASENAME
        BASENAME="$(basename "$EXTRA")"
        [ -f "$FW_DIR/$BASENAME" ] && codesign --force --sign - "$FW_DIR/$BASENAME"
    done

    info "Packaged $ARCH_NAME framework at $FW_DIR"
}

info "Packaging device framework..."
package_framework "ios-arm64" "$DEVICE_DYLIB" "$CONSTRAINT_DYLIB" "$METAL_DEVICE" "$TOPK_DEVICE"

info "Packaging simulator framework..."
package_framework "ios-arm64-simulator" "$SIM_DYLIB" "" "$METAL_SIM"

# ---------------------------------------------------------------------------
# 6. Create intermediate fat xcframework, then repackage
# ---------------------------------------------------------------------------
# The fat xcframework (with loose dylibs inside CLiteRTLM.framework) is
# rejected by App Store Connect (ITMS-90171). We produce it as an intermediate,
# then run repackage-xcframeworks.sh to split the loose dylibs into sibling
# xcframeworks (LiteRtMetalAccelerator, GemmaModelConstraintProvider) and fix
# CLiteRTLM's Info.plist + load commands.

info "Creating intermediate xcframework..."

rm -rf "$OUTPUT_DIR"

xcodebuild -create-xcframework \
    -framework "$WORK_DIR/ios-arm64/$FRAMEWORK_NAME.framework" \
    -framework "$WORK_DIR/ios-arm64-simulator/$FRAMEWORK_NAME.framework" \
    -output "$OUTPUT_DIR"

info "Repackaging into App-Store-compliant xcframeworks..."
"$SCRIPT_DIR/repackage-xcframeworks.sh"

# ---------------------------------------------------------------------------
# 7. Verify
# ---------------------------------------------------------------------------

info "Verifying xcframeworks..."

for XCF in LiteRTLM GemmaModelConstraintProvider LiteRtMetalAccelerator LiteRtTopKMetalSampler; do
    XCF_PATH="$PROJECT_DIR/Frameworks/$XCF.xcframework"
    [ -d "$XCF_PATH" ] || error "Missing $XCF_PATH"
    info "  $XCF.xcframework: $(du -sh "$XCF_PATH" | cut -f1)"
done

info "Done! Four xcframeworks ready under Frameworks/"
# WORK_DIR is cleaned up automatically by the EXIT trap
