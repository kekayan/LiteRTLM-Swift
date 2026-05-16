# LiteRTLM-Swift

Swift package for running [LiteRT-LM](https://ai.google.dev/edge/litert/lm) models on iOS. Wraps Google's C API in a clean, async/await Swift interface.

Supports **text generation**, **vision (image understanding)**, **audio (speech/sound understanding)**, and **streaming** with models like **Gemma 4 E2B**.

> **Note:** This is a community project, not an official Google product. The included `CLiteRTLM.xcframework` is built from Google's open-source [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) C API (Apache 2.0).

## Requirements

- iOS 17.0+
- Xcode 16+
- iPhone 13 Pro or later (6 GB+ RAM required for Gemma 4 E2B)
- `increased-memory-limit` entitlement (model loading needs ~4 GB RAM)

<details>
<summary>How to add the increased-memory-limit entitlement</summary>

In Xcode: select your app target > Signing & Capabilities > + Capability > search "Increased Memory Limit".

Or add manually to your `.entitlements` file:

```xml
<key>com.apple.developer.kernel.increased-memory-limit</key>
<true/>
```

Without this entitlement the system may kill your app during model loading.

</details>

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/kekayan/LiteRTLM-Swift.git", from: "0.1.0")
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            .product(name: "LiteRTLMSwift", package: "LiteRTLM-Swift")
        ]
    )
]
```

Or in Xcode: File > Add Package Dependencies > paste the repo URL > add `LiteRTLMSwift` to your target.

### GPU backend: install_name patch

If you pass `backend: "gpu"` to `LiteRTLMEngine`, add a **Run Script** build phase to your app target that runs after the frameworks are embedded. Without it, the engine's `dlopen("libLiteRtMetalAccelerator.dylib")` / `dlopen("libLiteRtTopKMetalSampler.dylib")` calls can miss the already-loaded Metal plugin frameworks and fall back to CPU paths.

**Why it's needed:** `ld -framework ...` requires framework binaries to use framework-style names, so the shipped xcframeworks contain `LiteRtMetalAccelerator` and `LiteRtTopKMetalSampler`. But the LiteRT-LM engine dlopens plugins by bare dylib leaf names. Patching the embedded binaries' install names to those leaves post-embed fixes the mismatch without breaking link-time framework discovery.

In the app target's **Build Phases** tab, add a new **Run Script Phase** after **Embed Frameworks** (declare the framework binary as an input so Xcode schedules it correctly):

**Input File:**
```
$(BUILT_PRODUCTS_DIR)/$(FRAMEWORKS_FOLDER_PATH)/LiteRtMetalAccelerator.framework/LiteRtMetalAccelerator
$(BUILT_PRODUCTS_DIR)/$(FRAMEWORKS_FOLDER_PATH)/LiteRtTopKMetalSampler.framework/LiteRtTopKMetalSampler
```

**Script:**
```sh
set -e
patch_id() {
  local fw="$1"
  local desired="$2"
  [ -f "$fw" ] || return 0
  [ "$(otool -D "$fw" | tail -1)" = "$desired" ] || install_name_tool -id "$desired" "$fw"
  codesign --force --sign "${EXPANDED_CODE_SIGN_IDENTITY:--}" "$fw"
}

patch_id "$BUILT_PRODUCTS_DIR/$FRAMEWORKS_FOLDER_PATH/LiteRtMetalAccelerator.framework/LiteRtMetalAccelerator" \
  "libLiteRtMetalAccelerator.dylib"
patch_id "$BUILT_PRODUCTS_DIR/$FRAMEWORKS_FOLDER_PATH/LiteRtTopKMetalSampler.framework/LiteRtTopKMetalSampler" \
  "libLiteRtTopKMetalSampler.dylib"
```

Tick **"Based on dependency analysis"** off (or set `alwaysOutOfDate = 1` in the pbxproj) so the phase always runs — the patch must survive an unpatched fresh embed after a clean build.

At launch, dyld resolves the app's pre-patch `LC_LOAD_DYLIB @rpath/...framework/...` entries via rpath, loads the files, and indexes each loaded image under the patched `LC_ID_DYLIB` leaf. When the engine later calls `dlopen("libLiteRtMetalAccelerator.dylib")` or `dlopen("libLiteRtTopKMetalSampler.dylib")`, dyld's install-name cache can answer with the existing handle.

If you only ever use `backend: "cpu"` (the default), you don't need this phase.

### Duplicate Obj-C class warnings

At launch you'll see `objc[...]: Class SRLRegistry is implemented in both ...LiteRt.framework... and ...LiteRtMetalAccelerator.framework...` (and similar for ~60 `GTM*`/`GIP*`/`SRL*`/`GSC*` classes across the three Metal-related frameworks).

These are benign in this build: every framework is compiled from the same upstream LiteRT-LM commit, so each duplicate class is a byte-identical statically-linked copy of the same Google internal logging/registry sources. The Obj-C runtime keeps the first-loaded copy and dispatches all calls through it; behavior is stable. The "may cause spurious casting failures" wording in the warning applies when duplicates have *differing* layouts — not the case here.

To silence the noise during development, add `OBJC_DISABLE_DUPLICATE_CLASS_CHECK = YES` to your scheme's **Run > Arguments > Environment Variables**. The variable only suppresses the warning print; it has no effect on App Store builds (where the scheme isn't used) and doesn't change runtime behavior.

## Quick Start

A complete end-to-end example:

```swift
import LiteRTLMSwift

// 1. Download model (~2.6 GB, only needed once)
let downloader = ModelDownloader()
try await downloader.download()  // defaults to Gemma 4 E2B from HuggingFace

// 2. Load engine
let engine = LiteRTLMEngine(modelPath: downloader.modelPath)
try await engine.load()  // takes ~5-10s on first launch

// 3. Generate text
let response = try await engine.generate(
    prompt: "<|turn>user\nWhat is Swift?\n<turn|>\n<|turn>model\n",
    temperature: 0.7,
    maxTokens: 256
)
print(response)

// 4. Vision (image understanding)
let imageData = try Data(contentsOf: photoURL)
let caption = try await engine.vision(
    imageData: imageData,  // JPEG, PNG, or HEIC
    prompt: "Describe this photo.",
    maxTokens: 512
)
print(caption)

// 5. Audio (speech/sound understanding)
let audioData = try Data(contentsOf: audioURL)
let transcript = try await engine.audio(
    audioData: audioData,  // WAV, FLAC, or MP3
    prompt: "Transcribe this audio.",
    maxTokens: 512
)
print(transcript)
```

> **Important:** Text generation (`generate`, `generateStreaming`, `openSession`) requires Gemma 4's turn marker format in the prompt (see [Prompt Format](#gemma-4-prompt-format)). All other methods — vision, audio, multimodal, and the persistent conversation API (`openConversation`/`conversationSend`) — take plain text prompts. The Conversation API handles formatting internally.

## More Examples

### Streaming

```swift
for try await chunk in engine.generateStreaming(
    prompt: "<|turn>user\nTell me a story.\n<turn|>\n<|turn>model\n"
) {
    print(chunk, terminator: "")
}
```

### Multi-Image Vision

```swift
let answer = try await engine.visionMultiImage(
    imagesData: [image1Data, image2Data],
    prompt: "Compare these two photos.",
    maxTokens: 1024
)
```

### Audio Understanding

Supports WAV, FLAC, and MP3. Audio is automatically resampled to 16 kHz mono internally.

```swift
let audioData = try Data(contentsOf: recordingURL)

// Transcription (default format: .wav)
let text = try await engine.audio(
    audioData: audioData,
    prompt: "Transcribe this audio."
)

// MP3 file
let mp3Data = try Data(contentsOf: mp3URL)
let summary = try await engine.audio(
    audioData: mp3Data,
    prompt: "Summarize what is being said.",
    format: .mp3,
    maxTokens: 1024
)
```

### Combined Audio + Vision (Multimodal)

Analyze audio and images together in a single query:

```swift
let response = try await engine.multimodal(
    audioData: [audioTrackData],
    imagesData: [keyframeData],
    prompt: "Does the speaker's description match what's shown in the image?"
)
```

### Multi-Turn Chat (KV Cache Reuse)

For multi-turn conversations, use the persistent session API. The KV cache is preserved across turns, reducing time-to-first-token from ~20s to ~1-2s on follow-up messages.

#### Text-only (Session API)

```swift
// Open a persistent session
try await engine.openSession(temperature: 0.7, maxTokens: 512)

// First turn — full prefill (~15-20s TTFT)
for try await chunk in engine.sessionGenerateStreaming(
    input: "<|turn>user\nHello!\n<turn|>\n<|turn>model\n"
) {
    print(chunk, terminator: "")
}

// Second turn — incremental prefill (~1-2s TTFT)
for try await chunk in engine.sessionGenerateStreaming(
    input: "<turn|>\n<|turn>user\nTell me more.\n<turn|>\n<|turn>model\n"
) {
    print(chunk, terminator: "")
}

// Clean up when done
engine.closeSession()
```

#### Multimodal (Conversation API)

Mix images, audio, and text freely across turns — each turn reuses the KV cache:

```swift
// Open a persistent multimodal conversation
try await engine.openConversation(temperature: 0.7)

// Turn 1: send an image
let description = try await engine.conversationSend(
    imagesData: [photoData],
    prompt: "What's in this photo?"
)

// Turn 2: text-only follow-up (KV cache reused — fast TTFT)
let detail = try await engine.conversationSend(
    prompt: "What color is the car in the background?"
)

// Turn 3: send audio
let answer = try await engine.conversationSend(
    audioData: [clipData],
    audioFormat: .wav,
    prompt: "Does this audio match the scene?"
)

// Clean up when done
engine.closeConversation()
```

### Tool Calling (Gemma 4)

Gemma 4 can emit structured tool calls. Declare tools with JSON-Schema `parameters`, open a conversation with them, and round-trip tool results back to the model. Constrained decoding is auto-enabled when you pass tools; pass `enableConstrainedDecoding: false` to opt out.

```swift
let getWeather = LiteRTLMTool(
    name: "get_weather",
    description: "Look up the current weather for a city.",
    parametersJSONSchema: """
    { "type": "object",
      "properties": { "city": { "type": "string", "description": "City name" } },
      "required": ["city"] }
    """
)

try await engine.openConversation(
    systemPrompt: "You are a helpful assistant.",
    tools: [getWeather],
    temperature: 0.2
)

switch try await engine.conversationSendTurn(prompt: "What's the weather in Tokyo?") {
case .text(let reply):
    print(reply)
case .toolCalls(let calls):
    // Execute tools locally, then send results back
    let results: [(String, [String: Any])] = calls.map { call in
        let city = call.arguments["city"]?.stringValue ?? "Unknown"
        return (call.name, ["temperature_c": 18, "conditions": "cloudy", "city": city])
    }
    switch try await engine.sendToolResultsTurn(results) {
    case .text(let final): print(final)
    case .toolCalls: print("Model requested more tool calls")
    }
}
```

### Thinking Mode (Gemma 4)

Enable thinking to see the model's reasoning before its final answer. Thoughts stream on a separate channel so the UI can present them distinctly. Past thoughts are auto-stripped from KV context on the next turn, so there's no cache-reuse penalty.

```swift
try await engine.openConversation(
    systemPrompt: "Solve problems step-by-step.",
    enableThinking: true
)

for try await event in engine.conversationSendTurnStreaming(
    prompt: "A train leaves at 3pm going 80 km/h. Another leaves 1h later at 100 km/h. When does it catch up?"
) {
    switch event {
    case .thought(let t): print("[thinking] \(t)", terminator: "")
    case .text(let text): print(text, terminator: "")
    case .toolCalls(let calls): print("tool calls: \(calls)")
    }
}
```

### Download Progress Tracking

`ModelDownloader` is `@Observable`, so you can bind directly in SwiftUI:

```swift
struct DownloadView: View {
    @State private var downloader = ModelDownloader()

    var body: some View {
        switch downloader.status {
        case .notStarted:
            Button("Download Model (\(downloader.totalBytesDisplay))") {
                Task { try await downloader.download() }
            }
        case .downloading(let progress):
            ProgressView(value: progress)
            Text("\(downloader.downloadedBytesDisplay) / \(downloader.totalBytesDisplay)")
            Button("Pause") { downloader.pause() }
        case .paused:
            Button("Resume") { Task { try await downloader.download() } }
        case .completed:
            Text("Model ready!")
        case .failed(let msg):
            Text("Error: \(msg)")
            Button("Retry") { Task { try await downloader.download() } }
        }
    }
}
```

### SwiftUI: Engine Status

```swift
struct EngineView: View {
    @State private var engine: LiteRTLMEngine

    init() {
        let path = ModelDownloader().modelPath
        _engine = State(initialValue: LiteRTLMEngine(modelPath: path))
    }

    var body: some View {
        Group {
            switch engine.status {
            case .notLoaded:
                Button("Load Model") { Task { try await engine.load() } }
            case .loading:
                ProgressView("Loading model...")
            case .ready:
                Text("Ready for inference!")
            case .error(let msg):
                Text("Error: \(msg)")
            }
        }
    }
}
```

## API Reference

### LiteRTLMEngine

| Method | Description |
|--------|-------------|
| `init(modelPath:backend:)` | Create engine. `backend`: `"cpu"` (default, recommended) or `"gpu"` (experimental, Metal) |
| `load()` | Load the `.litertlm` model. Call once, reuse across inferences |
| `unload()` | Free model memory |
| `generate(prompt:temperature:maxTokens:)` | One-shot text generation. Prompt must use Gemma turn markers |
| `generateStreaming(prompt:temperature:maxTokens:)` | Streaming text generation |
| `vision(imageData:prompt:temperature:maxTokens:maxImageDimension:)` | Single-image understanding. Plain text prompt |
| `visionMultiImage(imagesData:prompt:temperature:maxTokens:maxImageDimension:)` | Multi-image understanding |
| `audio(audioData:prompt:format:temperature:maxTokens:)` | Audio understanding (WAV, FLAC, MP3). Plain text prompt |
| `multimodal(audioData:audioFormat:imagesData:prompt:temperature:maxTokens:maxImageDimension:)` | Combined audio + vision inference |
| `openSession(temperature:maxTokens:)` | Open persistent text session for multi-turn chat (KV cache reuse) |
| `sessionGenerateStreaming(input:)` | Stream generation using persistent text session |
| `closeSession()` | Close persistent text session, free KV cache |
| `openConversation(temperature:maxTokens:)` | Open persistent multimodal conversation (KV cache reuse) |
| `openConversation(systemPrompt:tools:enableConstrainedDecoding:enableThinking:temperature:maxTokens:)` | Open conversation with typed tools and/or thinking mode (Gemma 4) |
| `conversationSend(audioData:audioFormat:imagesData:prompt:maxImageDimension:)` | Send a turn in the persistent conversation (any mix of audio/images/text) |
| `conversationSendTurn(prompt:)` | Send a turn and receive a typed `.text` / `.toolCalls` result |
| `conversationSendTurnStreaming(prompt:)` | Stream a turn as `LiteRTLMStreamEvent` (`.text` / `.thought` / `.toolCalls`) |
| `sendToolResultsTurn(_:)` | Send typed tool results and receive the follow-up turn |
| `cancelConversation()` | Cancel an in-flight conversation send |
| `closeConversation()` | Close persistent multimodal conversation, free KV cache |

| Property | Type | Description |
|----------|------|-------------|
| `status` | `Status` | `.notLoaded`, `.loading`, `.ready`, or `.error(String)` |
| `isReady` | `Bool` | Whether the engine is ready for inference |

### ModelDownloader

| Method | Description |
|--------|-------------|
| `init(modelsDirectory:)` | Create downloader. Default path: `~/Library/Application Support/LiteRTLM/Models/` |
| `download(from:)` | Download model from URL. Defaults to `defaultModelURL` (HuggingFace) |
| `pause()` | Pause download. Resume data is persisted to disk |
| `cancel()` | Cancel download and discard resume data |
| `deleteModel()` | Delete the downloaded model file |

| Property | Type | Description |
|----------|------|-------------|
| `status` | `DownloadStatus` | Current download state |
| `progress` | `Double` | 0.0 to 1.0 |
| `isDownloaded` | `Bool` | Whether the model file exists on disk |
| `modelPath` | `URL` | Full path to model file (use with `LiteRTLMEngine(modelPath:)`) |

### Gemma 4 Prompt Format

The **Session API** (text generation) requires Gemma 4's native turn marker format. The **Conversation API** (vision, audio, multimodal, persistent conversation) does NOT — just pass plain text.

```
<|turn>user
Your message here
<turn|>
<|turn>model
```

With system prompt:

```
<|turn>system
You are a helpful assistant.
<turn|>
<|turn>user
Hello!
<turn|>
<|turn>model
```

Multi-turn (for persistent session — only send the NEW content each turn):

```
# First turn input:
<|turn>user
Hello!
<turn|>
<|turn>model

# Second turn input (note the closing marker from previous model turn):
<turn|>
<|turn>user
Tell me more.
<turn|>
<|turn>model
```

## Architecture

```
┌──────────────────────────────────────────────┐
│                Your App                      │
├──────────────────────────────────────────────┤
│             LiteRTLMSwift                    │
│  ┌─────────────────┐  ┌──────────────────┐   │
│  │ LiteRTLMEngine  │  │ ModelDownloader  │   │
│  │                 │  │                  │   │
│  │ .generate()     │  │ .download()      │   │
│  │ .vision()       │  │ .pause()         │   │
│  │ .audio()        │  │ .cancel()        │   │
│  │ .multimodal()   │  │                  │   │
│  │ .openSession()  │  │                  │   │
│  │ .openConversation()                   │   │
│  └────────┬────────┘  └──────────────────┘   │
│           │                                  │
│     Serial DispatchQueue                     │
│     (thread safety)                          │
├───────────┼──────────────────────────────────┤
│     CLiteRTLM.xcframework (C API)            │
│           │                                  │
│   Session API          Conversation API      │
│   (text in/out)        (multimodal JSON)     │
│                                              │
│   For text generation  For vision / audio /  │
│   Raw prompt format    multimodal inference  │
└──────────────────────────────────────────────┘
```

- **Session API** — raw text prompts via `InputData`. You control the prompt format. Used by `generate()`, `generateStreaming()`, `openSession()`.
- **Conversation API** — JSON-based messages with image/audio file paths. Handles image decode/resize/patchify and audio decode/resample/mel-spectrogram internally. Used by `vision()`, `visionMultiImage()`, `audio()`, `multimodal()`, `openConversation()`/`conversationSend()`.
- All C API calls are serialized on a single `DispatchQueue` for thread safety. LiteRT-LM supports only one active session at a time.

## Building the XCFramework from Source

This repo ships prebuilt xcframeworks under `Frameworks/`. If you want to rebuild them yourself (e.g. to pick up upstream fixes or try a newer GPU backend), follow the steps below.

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Bazel | 7.6.1 | `brew install bazelisk` (auto-downloads correct version) |
| Git LFS | Latest | `brew install git-lfs` |
| Xcode | 16+ | Mac App Store |
| Disk space | ~20 GB | Bazel build cache |

### Option A: Build Script

```bash
# Clones LiteRT-LM source automatically and builds xcframework
./scripts/build-xcframework.sh

# Build a specific upstream ref
LITERT_LM_REF=v0.11.0 ./scripts/build-xcframework.sh

# Or point to an existing local checkout
./scripts/build-xcframework.sh ~/Dev/LiteRT-LM
```

The script will:
1. Clone (or use existing) [google-ai-edge/LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) source
2. Patch `c/BUILD` if needed — adds the `cc_binary` dylib target (missing in published releases through v0.11.0) and stubs `ios_engine.bzl` (missing in some HEAD snapshots)
3. Build `libLiteRTLMEngine.dylib` for `ios_arm64` (device) and `ios_sim_arm64` (simulator)
4. Pull the upstream iOS prebuilts from Git LFS (`LiteRt`, Metal accelerator, TopK sampler)
5. Package App-Store-compliant sibling xcframeworks under `Frameworks/`

You can also run the **Build XCFrameworks** GitHub Action manually. It checks out the requested upstream LiteRT-LM ref, pulls the iOS prebuilt LFS objects, runs the same build script, verifies the package manifest, and uploads the generated xcframeworks as an artifact.

### Option B: Manual Step-by-Step

#### 1. Clone LiteRT-LM source

```bash
git clone https://github.com/google-ai-edge/LiteRT-LM.git
cd LiteRT-LM
```

#### 2. Patch `c/BUILD`

The upstream BUILD file may need patching depending on the version:

**a) Published releases through v0.11.0** — the `cc_binary` target for the shared library doesn't exist yet. Append it:

```bash
cat >> c/BUILD << 'EOF'

cc_binary(
    name = "libLiteRTLMEngine.dylib",
    srcs = [
        "engine.cc",
        "engine.h",
        "litert_lm_logging.cc",
        "litert_lm_logging.h",
    ],
    linkopts = [
        "-Wl,-exported_symbol,_litert_lm_*",
    ],
    linkshared = True,
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = ENGINE_COMMON_DEPS + [
        "//runtime/core:engine_impl",
    ],
)
EOF
```

**b) Some upstream HEAD snapshots** — `c/BUILD` loads `ios_engine.bzl` which isn't published. Create a stub:

```bash
cat > c/ios_engine.bzl << 'EOF'
"""Stub for ios_shared_engine macro (not yet published upstream)."""

def ios_shared_engine(**kwargs):
    pass
EOF
```

> The build script (`Option A`) detects and applies both patches automatically.

#### 3. Build for iOS device (arm64)

```bash
bazel build --config=ios_arm64 //c:libLiteRTLMEngine.dylib
```

Output: `bazel-bin/c/libLiteRTLMEngine.dylib`

The Bazel build target is defined in [`c/BUILD`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/c/BUILD):
- `linkshared = True` + `linkstatic = True` — produces a self-contained dylib with all C++ deps statically linked
- `-Wl,-exported_symbol,_litert_lm_*` — only exports the public C API symbols

#### 4. Build for iOS simulator (arm64)

```bash
# Save device dylib first (Bazel overwrites bazel-bin between configs)
cp bazel-bin/c/libLiteRTLMEngine.dylib /tmp/libLiteRTLMEngine-device.dylib

bazel build --config=ios_sim_arm64 //c:libLiteRTLMEngine.dylib
cp bazel-bin/c/libLiteRTLMEngine.dylib /tmp/libLiteRTLMEngine-sim.dylib
```

Available iOS configs in `.bazelrc`:

| Config | Architecture | Use Case |
|--------|-------------|----------|
| `ios_arm64` | arm64 | Physical device |
| `ios_sim_arm64` | arm64 | Apple Silicon simulator |
| `ios_x86_64` | x86_64 | Intel Mac simulator |
| `ios_arm64e` | arm64e | A12+ with pointer auth |

#### 5. Package as .framework bundles

Each architecture needs to be wrapped in a `.framework` bundle before creating the xcframework.

```bash
# Device framework
mkdir -p /tmp/ios-arm64/CLiteRTLM.framework/{Headers,Modules}
cp /tmp/libLiteRTLMEngine-device.dylib /tmp/ios-arm64/CLiteRTLM.framework/CLiteRTLM
install_name_tool -id "@rpath/CLiteRTLM.framework/CLiteRTLM" /tmp/ios-arm64/CLiteRTLM.framework/CLiteRTLM

# Simulator framework
mkdir -p /tmp/ios-arm64-simulator/CLiteRTLM.framework/{Headers,Modules}
cp /tmp/libLiteRTLMEngine-sim.dylib /tmp/ios-arm64-simulator/CLiteRTLM.framework/CLiteRTLM
install_name_tool -id "@rpath/CLiteRTLM.framework/CLiteRTLM" /tmp/ios-arm64-simulator/CLiteRTLM.framework/CLiteRTLM
```

Copy headers (from the LiteRT-LM source `c/` directory):

```bash
for DIR in /tmp/ios-arm64 /tmp/ios-arm64-simulator; do
    cp c/engine.h "$DIR/CLiteRTLM.framework/Headers/"
    cp c/litert_lm_logging.h "$DIR/CLiteRTLM.framework/Headers/"
done
```

Create `module.modulemap` (same for both):

```bash
for DIR in /tmp/ios-arm64 /tmp/ios-arm64-simulator; do
    cat > "$DIR/CLiteRTLM.framework/Modules/module.modulemap" << 'EOF'
framework module CLiteRTLM {
    header "engine.h"
    export *
}
EOF
done
```

Create `Info.plist` (same for both):

```bash
for DIR in /tmp/ios-arm64 /tmp/ios-arm64-simulator; do
    cat > "$DIR/CLiteRTLM.framework/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>CLiteRTLM</string>
    <key>CFBundleIdentifier</key>
    <string>com.google.CLiteRTLM</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>CLiteRTLM</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>MinimumOSVersion</key>
    <string>13.0</string>
</dict>
</plist>
EOF
done
```

Ad-hoc code sign:

```bash
codesign --force --sign - /tmp/ios-arm64/CLiteRTLM.framework/CLiteRTLM
codesign --force --sign - /tmp/ios-arm64-simulator/CLiteRTLM.framework/CLiteRTLM
```

#### 6. Create the xcframework

```bash
xcodebuild -create-xcframework \
    -framework /tmp/ios-arm64/CLiteRTLM.framework \
    -framework /tmp/ios-arm64-simulator/CLiteRTLM.framework \
    -output Frameworks/LiteRTLM.xcframework
```

#### 7. Verify

```bash
# Check architectures
file Frameworks/LiteRTLM.xcframework/ios-arm64/CLiteRTLM.framework/CLiteRTLM
# -> Mach-O 64-bit dynamically linked shared library arm64

file Frameworks/LiteRTLM.xcframework/ios-arm64-simulator/CLiteRTLM.framework/CLiteRTLM
# -> Mach-O 64-bit dynamically linked shared library arm64 (simulator)

# Check exported symbols
nm -gU Frameworks/LiteRTLM.xcframework/ios-arm64/CLiteRTLM.framework/CLiteRTLM | grep litert_lm
# Should list all litert_lm_* public API functions
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `no such target '//c:libLiteRTLMEngine.dylib'` | Two possible causes: (1) published releases through v0.11.0 don't define the `cc_binary` dylib target — append it per Step 2a; (2) some HEAD snapshots load a missing `ios_engine.bzl` which breaks BUILD parsing — create the stub per Step 2b |
| `no such package '@build_bazel_apple_support'` | Run `bazel sync` to fetch external dependencies |
| Xcode SDK not found | Ensure Xcode is selected: `sudo xcode-select -s /Applications/Xcode.app` |
| Build takes very long | First build downloads ~10 GB of deps. Subsequent builds use cache |
| `Undefined symbols` at link time | Make sure you're using `//c:libLiteRTLMEngine.dylib` target, not `//c:engine` |
| Code signing errors | Use ad-hoc signing (`--sign -`) for development; real signing happens at app archive |

## License

MIT License. See [LICENSE](LICENSE).

The `CLiteRTLM.xcframework` contains code from Google's LiteRT-LM project, licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
