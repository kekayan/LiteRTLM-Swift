#!/usr/bin/env bash
#
# Repackage the fat LiteRTLM.xcframework into three App-Store-compliant
# xcframeworks:
#
#   1. LiteRTLM.xcframework                     — CLiteRTLM only (no loose dylibs)
#   2. GemmaModelConstraintProvider.xcframework — sibling framework for the Gemma
#                                                 constraint provider dylib
#   3. LiteRtMetalAccelerator.xcframework       — sibling framework for the Metal
#                                                 accelerator dylib (dlopen'd at
#                                                 runtime by the engine)
#
# The fix addresses App Store Connect errors:
#   - ITMS-90171: loose .dylib files inside .framework are rejected
#   - ITMS-90057: CFBundleShortVersionString missing from CLiteRTLM Info.plist
#
# Runtime details:
#   - CLiteRTLM hard-links GemmaModelConstraintProvider via LC_LOAD_DYLIB.
#     We rewrite it to @rpath/GemmaModelConstraintProvider.framework/<binary>
#     and add LC_RPATH @loader_path/.. so dyld finds it at app's Frameworks/.
#   - The Metal dylib is dlopen'd by leaf name. We set its install_name to the
#     bare leaf "libLiteRtMetalAccelerator.dylib"; LiteRTLMSwift preemptively
#     dlopens the framework by full path, after which dyld's install-name cache
#     answers the engine's leaf-name dlopen.
#
# Input:  Frameworks/LiteRTLM.xcframework (as produced by build-xcframework.sh
#         or its slice-builder mode, see --slices flag).
# Output: Frameworks/{LiteRTLM,GemmaModelConstraintProvider,LiteRtMetalAccelerator}.xcframework

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
set_ios_min() {
    local BIN="$1" PLATFORM="$2"  # PLATFORM: ios or iossim
    xcrun vtool -set-build-version "$PLATFORM" 17.0 26.2 -replace -output "$BIN" "$BIN" >/dev/null
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
METAL_DEVICE_DYLIB="$DEVICE_SRC/libLiteRtMetalAccelerator.dylib"
[ -f "$METAL_DEVICE_DYLIB" ] || METAL_DEVICE_DYLIB="$(find_existing_metal ios-arm64)"
METAL_SIM_DYLIB="$SIM_SRC/libLiteRtMetalAccelerator.dylib"
[ -f "$METAL_SIM_DYLIB" ] || METAL_SIM_DYLIB="$(find_existing_metal ios-arm64-simulator)"
[ -f "$GEMMA_DEVICE_DYLIB" ] || error "Gemma device dylib not found"
[ -f "$METAL_DEVICE_DYLIB" ] || error "Metal device dylib not found"
[ -f "$METAL_SIM_DYLIB" ]    || error "Metal sim dylib not found"

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
    <key>CFBundleExecutable</key><string>$EXEC</string>
    <key>CFBundleIdentifier</key><string>$IDENT</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleName</key><string>$NAME</string>
    <key>CFBundlePackageType</key><string>FMWK</string>
    <key>CFBundleShortVersionString</key><string>1.0.0</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>MinimumOSVersion</key><string>13.0</string>
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
          "$DEST"/libGemmaModelConstraintProvider.dylib
    rm -rf "$DEST/_CodeSignature"
    write_plist "$DEST" "CLiteRTLM" "com.google.CLiteRTLM"
    if [ "$IS_DEVICE" = "yes" ]; then
        install_name_tool -change \
            "@rpath/libGemmaModelConstraintProvider.dylib" \
            "@rpath/GemmaModelConstraintProvider.framework/GemmaModelConstraintProvider" \
            "$DEST/CLiteRTLM"
        # Add @loader_path/.. so sibling frameworks resolve from app Frameworks/.
        # The existing @loader_path rpath stays (harmless; nothing resolves there now).
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
    local BIN_NAME="libLiteRtMetalAccelerator.dylib"
    mkdir -p "$DEST"
    # The engine (gpu_registry.cc) calls dlopen("libLiteRtMetalAccelerator.dylib")
    # with a bare leaf and rejects the GPU backend if that dlopen fails.
    # dyld's leaf-name dlopen matches against the LEAF of each loaded image's
    # install_name — so the binary inside the framework is named exactly that,
    # CFBundleExecutable matches, and install_name's leaf matches. Then the
    # engine's leaf-name dlopen cache-hits and GPU registration succeeds.
    cp "$SRC_DYLIB" "$DEST/$BIN_NAME"
    install_name_tool -id \
        "@rpath/LiteRtMetalAccelerator.framework/$BIN_NAME" \
        "$DEST/$BIN_NAME"
    case "$SLICE" in
        ios-arm64)           set_ios_min "$DEST/$BIN_NAME" ios ;;
        ios-arm64-simulator) set_ios_min "$DEST/$BIN_NAME" iossim ;;
    esac
    write_plist "$DEST" "LiteRtMetalAccelerator" "com.google.LiteRtMetalAccelerator" "$BIN_NAME"
    codesign --force --sign - "$DEST/$BIN_NAME"
    info "Built Metal slice: $SLICE"
}

make_metal_slice "ios-arm64" "$METAL_DEVICE_DYLIB"
make_metal_slice "ios-arm64-simulator" "$METAL_SIM_DYLIB"

# ---------------------------------------------------------------------------
# 4. Create the three xcframeworks
# ---------------------------------------------------------------------------

rm -rf "$FRAMEWORKS_DIR/LiteRTLM.xcframework" \
       "$FRAMEWORKS_DIR/GemmaModelConstraintProvider.xcframework" \
       "$FRAMEWORKS_DIR/LiteRtMetalAccelerator.xcframework"

xcodebuild -create-xcframework \
    -framework "$WORK_DIR/core/ios-arm64/CLiteRTLM.framework" \
    -framework "$WORK_DIR/core/ios-arm64-simulator/CLiteRTLM.framework" \
    -output "$FRAMEWORKS_DIR/LiteRTLM.xcframework" >/dev/null

xcodebuild -create-xcframework \
    -framework "$WORK_DIR/gemma/ios-arm64/GemmaModelConstraintProvider.framework" \
    -framework "$WORK_DIR/gemma/ios-arm64-simulator/GemmaModelConstraintProvider.framework" \
    -output "$FRAMEWORKS_DIR/GemmaModelConstraintProvider.xcframework" >/dev/null

# Manually assemble the Metal xcframework because `xcodebuild -create-xcframework`
# expects the framework binary's filename to equal the framework name (without
# .framework) — but our Metal framework is intentionally named
# "libLiteRtMetalAccelerator.dylib" so that dyld's leaf-name dlopen match works.
MOUT="$FRAMEWORKS_DIR/LiteRtMetalAccelerator.xcframework"
mkdir -p "$MOUT/ios-arm64" "$MOUT/ios-arm64-simulator"
cp -R "$WORK_DIR/metal/ios-arm64/LiteRtMetalAccelerator.framework" "$MOUT/ios-arm64/"
cp -R "$WORK_DIR/metal/ios-arm64-simulator/LiteRtMetalAccelerator.framework" "$MOUT/ios-arm64-simulator/"
cat > "$MOUT/Info.plist" << 'XFWK_PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>AvailableLibraries</key>
	<array>
		<dict>
			<key>BinaryPath</key>
			<string>LiteRtMetalAccelerator.framework/libLiteRtMetalAccelerator.dylib</string>
			<key>LibraryIdentifier</key>
			<string>ios-arm64</string>
			<key>LibraryPath</key>
			<string>LiteRtMetalAccelerator.framework</string>
			<key>SupportedArchitectures</key>
			<array>
				<string>arm64</string>
			</array>
			<key>SupportedPlatform</key>
			<string>ios</string>
		</dict>
		<dict>
			<key>BinaryPath</key>
			<string>LiteRtMetalAccelerator.framework/libLiteRtMetalAccelerator.dylib</string>
			<key>LibraryIdentifier</key>
			<string>ios-arm64-simulator</string>
			<key>LibraryPath</key>
			<string>LiteRtMetalAccelerator.framework</string>
			<key>SupportedArchitectures</key>
			<array>
				<string>arm64</string>
			</array>
			<key>SupportedPlatform</key>
			<string>ios</string>
			<key>SupportedPlatformVariant</key>
			<string>simulator</string>
		</dict>
	</array>
	<key>CFBundlePackageType</key>
	<string>XFWK</string>
	<key>XCFrameworkFormatVersion</key>
	<string>1.0</string>
</dict>
</plist>
XFWK_PLIST

info "Wrote:"
info "  $FRAMEWORKS_DIR/LiteRTLM.xcframework"
info "  $FRAMEWORKS_DIR/GemmaModelConstraintProvider.xcframework"
info "  $FRAMEWORKS_DIR/LiteRtMetalAccelerator.xcframework"
