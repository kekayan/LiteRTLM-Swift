import Foundation
import CoreGraphics
import ImageIO
import os
import CLiteRTLM

/// Swift wrapper for Google's LiteRT-LM on-device inference engine.
///
/// Supports text generation (Session API) and multimodal inference — vision
/// and audio — (Conversation API) with `.litertlm` model files (e.g. Gemma 4 E2B).
///
/// Thread safety: all C API calls are serialized on an internal dispatch queue.
/// The class is `@unchecked Sendable` because OpaquePointers are only accessed
/// on that queue.
///
/// ## Quick Start
/// ```swift
/// let engine = LiteRTLMEngine(modelPath: modelURL)
/// try await engine.load()
///
/// // Text
/// let response = try await engine.generate(prompt: "Hello!", temperature: 0.7, maxTokens: 256)
///
/// // Vision
/// let caption = try await engine.vision(imageData: jpegData, prompt: "Describe this photo.")
///
/// // Audio
/// let transcript = try await engine.audio(audioData: wavData, prompt: "Transcribe this audio.")
/// ```
@Observable
public final class LiteRTLMEngine: @unchecked Sendable {

    // MARK: - Types

    public enum Status: Sendable, Equatable {
        case notLoaded
        case loading
        case ready
        case error(String)
    }

    // MARK: - Properties

    public private(set) var status: Status = .notLoaded

    /// Whether the engine is ready for inference (text, vision, and audio).
    public var isReady: Bool { status == .ready }

    private let modelPath: URL
    private let backend: String
    /// When `nil`, derived in `load()` from `backend` (GPU → `"gpu"`).
    private let visionBackend: String?
    /// When `nil`, derived in `load()` from `backend` (GPU → `"cpu"` for Gemma E2B audio adapter constraints).
    private let audioBackend: String?

    private var engine: OpaquePointer?  // LiteRtLmEngine*
    /// `.default` QoS is deliberate: the streaming paths below block on a
    /// `DispatchSemaphore` that the LiteRT-LM C library signals from its own
    /// worker thread (observed at `.default` QoS). Running this queue at
    /// `.userInitiated` triggers a runtime priority-inversion warning each
    /// time a stream starts. The dispatch queue is only a serialization
    /// point — actual inference runs on the C library's internal threads —
    /// so dropping to `.default` has no measurable latency effect.
    private let inferenceQueue = DispatchQueue(label: "com.litertlm.inference", qos: .default)

    private static let log = Logger(subsystem: "LiteRTLMSwift", category: "Engine")

    // Belt-and-braces Metal plugin preload. The frameworks auto-load at launch
    // via LC_LOAD_DYLIB so their symbols are in the process namespace; these
    // dlopen calls are no-ops in that case. They matter if a consumer ever
    // configures a framework as "embed without link" (no LC_LOAD_DYLIB).
    private static let preloadPlugins: Void = {
        guard let frameworksPath = Bundle.main.privateFrameworksPath else { return }

        let plugins = [
            ("LiteRtMetalAccelerator", "LiteRtMetalAccelerator"),
            ("LiteRtTopKMetalSampler", "LiteRtTopKMetalSampler"),
        ]

        for (frameworkName, executableName) in plugins {
            let pluginPath = "\(frameworksPath)/\(frameworkName).framework/\(executableName)"
            guard FileManager.default.fileExists(atPath: pluginPath) else {
                log.debug("\(frameworkName, privacy: .public) framework not present at \(pluginPath, privacy: .public); related GPU features may fall back")
                continue
            }
            if dlopen(pluginPath, RTLD_NOW | RTLD_GLOBAL) == nil {
                let err = dlerror().map { String(cString: $0) } ?? "unknown"
                log.error("dlopen \(frameworkName, privacy: .public) failed: \(err, privacy: .public)")
            }
        }
    }()

    // MARK: - Init

    /// Create an engine instance.
    /// - Parameters:
    ///   - modelPath: Path to the `.litertlm` model file on disk.
    ///   - backend: Main LM backend — `"cpu"` or `"gpu"` (GPU uses Metal on iOS).
    ///   - visionBackend: Vision encoder backend, or `nil` to default **`"cpu"`** (including when
    ///     `backend` is `"gpu"`). Gemma E2B vision graphs are often CPU-only in LiteRT constraints; defaulting
    ///     vision to GPU caused `litert_lm_engine_create` failures on device while AI Edge Gallery still
    ///     showed “GPU” for the main LM. Pass `"gpu"` explicitly if your model supports it.
    ///   - audioBackend: Audio adapter backend, or `nil` to default **`"cpu"`** (Gemma E2B audio is CPU-only;
    ///     avoids `Audio backend constraint mismatch` when the main backend is `"gpu"`).
    public init(
        modelPath: URL,
        backend: String = "cpu",
        visionBackend: String? = nil,
        audioBackend: String? = nil
    ) {
        self.modelPath = modelPath
        self.backend = backend
        self.visionBackend = visionBackend
        self.audioBackend = audioBackend
    }

    deinit {
        let eng = engine
        let ses = chatSession
        let sesCfg = chatSessionConfig
        let conv = multimodalConversation
        let convCfg = multimodalConvConfig
        let convSesCfg = multimodalSessionConfig
        let queue = inferenceQueue
        if eng != nil || ses != nil || conv != nil {
            queue.async {
                if let s = ses { litert_lm_session_delete(s) }
                if let c = sesCfg { litert_lm_session_config_delete(c) }
                if let c = conv { litert_lm_conversation_delete(c) }
                if let c = convCfg { litert_lm_conversation_config_delete(c) }
                if let c = convSesCfg { litert_lm_session_config_delete(c) }
                if let e = eng { litert_lm_engine_delete(e) }
            }
        }
    }

    // MARK: - Lifecycle

    /// Load the `.litertlm` model. Call once, reuse for multiple inferences.
    /// Vision and audio encoders are embedded in the model file — no separate load step needed.
    ///
    /// `maxNumTokens` sizes the engine's KV cache (total context window:
    /// system + tools + history + prompt + decode). The runtime substitutes
    /// this value into the model's magic-number tensor shapes, so it must
    /// be set at load time and cannot grow afterwards. Defaults to 4096;
    /// callers should pass the model's declared context window (e.g.
    /// Gemma 4 E2B-it: 32K) clamped to whatever the device can hold —
    /// every doubling roughly doubles KV cache memory.
    @MainActor
    public func load(maxNumTokens: Int32 = 4096) async throws {
        guard status != .ready && status != .loading else { return }

        _ = Self.preloadPlugins

        status = .loading

        let path = modelPath.path
        let backendStr = self.backend
        let visionStr = visionBackend?.lowercased() ?? "cpu"
        let audioStr = audioBackend?.lowercased() ?? "cpu"
        Self.log.info(
            "Loading model: \(self.modelPath.lastPathComponent), backend: \(self.backend) (vision: \(visionStr), audio: \(audioStr))"
        )

        let startTime = CFAbsoluteTimeGetCurrent()

        guard FileManager.default.fileExists(atPath: path) else {
            let msg = "Model file not found at \(path)"
            Self.log.error("\(msg)")
            status = .error(msg)
            throw LiteRTLMError.modelNotFound
        }

        do {
            let createdEngine = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<OpaquePointer, any Error>) in
                self.inferenceQueue.async {
                    do {
                        litert_lm_set_min_log_level(1)

                        guard let settings = litert_lm_engine_settings_create(
                            path, backendStr, visionStr, audioStr
                        ) else {
                            throw LiteRTLMError.engineCreationFailed("Failed to create engine settings")
                        }

                        litert_lm_engine_settings_set_max_num_tokens(settings, maxNumTokens)

                        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
                            .appendingPathComponent("litertlm_cache").path
                        try? FileManager.default.createDirectory(atPath: cacheDir, withIntermediateDirectories: true)
                        litert_lm_engine_settings_set_cache_dir(settings, cacheDir)

                        litert_lm_engine_settings_enable_benchmark(settings)

                        // litert_lm_engine_create compiles and loads the model
                        // in-place and needs more than GCD's default 512KB stack.
                        // Spin a dedicated thread with 8MB stack; block until done.
                        let engineSem = DispatchSemaphore(value: 0)
                        var rawEngine: OpaquePointer? = nil
                        let t = Thread { rawEngine = litert_lm_engine_create(settings); engineSem.signal() }
                        t.stackSize = 8 * 1024 * 1024
                        t.start()
                        engineSem.wait()
                        litert_lm_engine_settings_delete(settings)

                        guard let createdEngine = rawEngine else {
                            throw LiteRTLMError.engineCreationFailed("litert_lm_engine_create returned NULL")
                        }

                        continuation.resume(returning: createdEngine)
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }

            inferenceQueue.sync { self.engine = createdEngine }

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            Self.log.info("Model loaded in \(String(format: "%.1f", elapsed))s")
            status = .ready
        } catch {
            let msg = "Load failed: \(error.localizedDescription)"
            Self.log.error("\(msg)")
            status = .error(msg)
            throw error
        }
    }

    /// Unload the model to free memory.
    @MainActor
    public func unload() {
        inferenceQueue.sync {
            if let s = chatSession {
                litert_lm_session_delete(s)
                chatSession = nil
            }
            if let c = chatSessionConfig {
                litert_lm_session_config_delete(c)
                chatSessionConfig = nil
            }
            if let c = multimodalConversation {
                litert_lm_conversation_delete(c)
                multimodalConversation = nil
            }
            if let c = multimodalConvConfig {
                litert_lm_conversation_config_delete(c)
                multimodalConvConfig = nil
            }
            if let c = multimodalSessionConfig {
                litert_lm_session_config_delete(c)
                multimodalSessionConfig = nil
            }
            if let eng = engine { litert_lm_engine_delete(eng) }
            engine = nil
        }
        status = .notLoaded
        Self.log.info("Model unloaded")
    }

    // MARK: - Text Generation (Session API)

    /// Generate text from a prompt. Creates a one-shot session per call.
    ///
    /// - Parameters:
    ///   - prompt: The input text. For Gemma 4, use `<|turn>user\n...<turn|>\n<|turn>model\n` format.
    ///   - temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative). Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 512.
    /// - Returns: Generated text.
    public func generate(
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 512
    ) async throws -> String {
        try ensureReady()
        return try await runSessionInference(
            prompt: prompt, temperature: temperature, maxTokens: Int32(maxTokens)
        )
    }

    /// Stream text generation token by token.
    ///
    /// Creates a one-shot session per call. For multi-turn conversations with
    /// KV cache reuse, use the persistent session API instead.
    ///
    /// - Parameters:
    ///   - prompt: The input text.
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 512.
    /// - Returns: An `AsyncThrowingStream` yielding text chunks.
    public func generateStreaming(
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 512
    ) -> AsyncThrowingStream<String, Error> {
        runSessionInferenceStreaming(
            prompt: prompt, temperature: temperature, maxTokens: Int32(maxTokens)
        )
    }

    // MARK: - Vision (Conversation API)

    /// Run vision inference on a single image.
    ///
    /// Uses the Conversation API, which handles image decoding, resizing, and
    /// patchification internally. Input images are auto-converted to JPEG and
    /// resized to fit within `maxImageDimension`.
    ///
    /// - Parameters:
    ///   - imageData: Raw image bytes (JPEG, PNG, HEIC, etc.).
    ///   - prompt: Text prompt for the vision model (e.g., "Describe this photo.").
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 512.
    ///   - maxImageDimension: Resize long edge to this value. Default 1024.
    /// - Returns: Generated text response.
    public func vision(
        imageData: Data,
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 512,
        maxImageDimension: Int = 1024
    ) async throws -> String {
        try ensureReady()

        guard let jpegData = Self.prepareImageForVision(imageData, maxDimension: maxImageDimension) else {
            throw LiteRTLMError.inferenceFailure("Failed to convert image to JPEG")
        }

        let tempURL = Self.makeTempURL(extension: "jpg")
        try jpegData.write(to: tempURL)

        let messageJSON = Self.buildMultimodalMessageJSON(
            audioPaths: [], imagePaths: [tempURL.path], text: prompt
        )
        return try await runConversationInference(
            messageJSON: messageJSON,
            tempURLs: [tempURL],
            temperature: temperature,
            maxTokens: maxTokens
        )
    }

    /// Run vision inference on multiple images.
    ///
    /// - Parameters:
    ///   - imagesData: Array of raw image bytes.
    ///   - prompt: Text prompt about the images.
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 1024.
    ///   - maxImageDimension: Resize long edge to this value. Default 1024.
    /// - Returns: Generated text response.
    public func visionMultiImage(
        imagesData: [Data],
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 1024,
        maxImageDimension: Int = 1024
    ) async throws -> String {
        try ensureReady()
        guard !imagesData.isEmpty else {
            throw LiteRTLMError.inferenceFailure("No images provided")
        }

        var tempURLs: [URL] = []
        do {
            for (i, data) in imagesData.enumerated() {
                guard let jpegData = Self.prepareImageForVision(data, maxDimension: maxImageDimension) else {
                    throw LiteRTLMError.inferenceFailure("Failed to convert image \(i + 1) to JPEG")
                }
                let url = Self.makeTempURL(extension: "jpg")
                try jpegData.write(to: url)
                tempURLs.append(url)
            }
        } catch {
            Self.cleanupTempFiles(tempURLs)
            throw error
        }

        let messageJSON = Self.buildMultimodalMessageJSON(
            audioPaths: [], imagePaths: tempURLs.map(\.path), text: prompt
        )
        return try await runConversationInference(
            messageJSON: messageJSON,
            tempURLs: tempURLs,
            temperature: temperature,
            maxTokens: maxTokens
        )
    }

    // MARK: - Audio (Conversation API)

    /// Supported audio formats for the `audio()` and `multimodal()` methods.
    public enum AudioFormat: String, Sendable {
        case wav, flac, mp3
    }

    /// Run audio inference on a single audio file.
    ///
    /// Uses the Conversation API, which handles audio decoding and preprocessing
    /// (resample to 16 kHz, convert to mel spectrogram) internally.
    ///
    /// - Parameters:
    ///   - audioData: Raw audio bytes (WAV, FLAC, or MP3).
    ///   - prompt: Text prompt (e.g., "Transcribe this audio.", "Summarize what is being said.").
    ///   - format: Audio container format. Default `.wav`.
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 512.
    /// - Returns: Generated text response.
    public func audio(
        audioData: Data,
        prompt: String,
        format: AudioFormat = .wav,
        temperature: Float = 0.7,
        maxTokens: Int = 512
    ) async throws -> String {
        try ensureReady()
        guard !audioData.isEmpty else {
            throw LiteRTLMError.inferenceFailure("No audio data provided")
        }

        let tempURL = Self.makeTempURL(extension: format.rawValue)
        try audioData.write(to: tempURL)

        let messageJSON = Self.buildMultimodalMessageJSON(
            audioPaths: [tempURL.path], imagePaths: [], text: prompt
        )
        return try await runConversationInference(
            messageJSON: messageJSON,
            tempURLs: [tempURL],
            temperature: temperature,
            maxTokens: maxTokens
        )
    }

    /// Run multimodal inference combining audio, images, and text in a single query.
    ///
    /// Useful for tasks like "describe what's happening in this video" where you have
    /// both the audio track and keyframes, or "does this photo match what the speaker describes?".
    ///
    /// - Parameters:
    ///   - audioData: Array of raw audio bytes (WAV, FLAC, or MP3). Pass empty array to skip.
    ///   - imagesData: Array of raw image bytes (JPEG, PNG, HEIC). Pass empty array to skip.
    ///   - prompt: Text prompt about the audio and/or images.
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 1024.
    ///   - maxImageDimension: Resize image long edge to this value. Default 1024.
    /// - Returns: Generated text response.
    public func multimodal(
        audioData: [Data] = [],
        audioFormat: AudioFormat = .wav,
        imagesData: [Data] = [],
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 1024,
        maxImageDimension: Int = 1024
    ) async throws -> String {
        try ensureReady()
        guard !audioData.isEmpty || !imagesData.isEmpty else {
            throw LiteRTLMError.inferenceFailure("No audio or image data provided")
        }

        var tempURLs: [URL] = []
        var audioPaths: [String] = []
        var imagePaths: [String] = []

        do {
            // Write audio files
            for (i, data) in audioData.enumerated() {
                guard !data.isEmpty else {
                    throw LiteRTLMError.inferenceFailure("Audio data \(i + 1) is empty")
                }
                let url = Self.makeTempURL(extension: audioFormat.rawValue)
                try data.write(to: url)
                tempURLs.append(url)
                audioPaths.append(url.path)
            }

            // Write image files
            for (i, data) in imagesData.enumerated() {
                guard let jpegData = Self.prepareImageForVision(data, maxDimension: maxImageDimension) else {
                    throw LiteRTLMError.inferenceFailure("Failed to convert image \(i + 1) to JPEG")
                }
                let url = Self.makeTempURL(extension: "jpg")
                try jpegData.write(to: url)
                tempURLs.append(url)
                imagePaths.append(url.path)
            }
        } catch {
            Self.cleanupTempFiles(tempURLs)
            throw error
        }

        let messageJSON = Self.buildMultimodalMessageJSON(
            audioPaths: audioPaths, imagePaths: imagePaths, text: prompt
        )
        return try await runConversationInference(
            messageJSON: messageJSON,
            tempURLs: tempURLs,
            temperature: temperature,
            maxTokens: maxTokens
        )
    }

    // MARK: - Persistent Session (KV Cache Reuse)
    //
    // LiteRT-LM's Session maintains a KV cache across multiple generate_content
    // calls. By keeping the session alive across turns, subsequent messages only
    // need to prefill NEW tokens instead of the entire conversation history.
    // This reduces TTFT from ~20s (full prefill) to ~1-2s (incremental).

    private var chatSession: OpaquePointer?
    private var chatSessionConfig: OpaquePointer?

    /// Open a persistent session for multi-turn generation with KV cache reuse.
    ///
    /// Call once when a conversation begins. Subsequent calls to
    /// `sessionGenerateStreaming(input:)` reuse this session's KV cache.
    ///
    /// - Parameters:
    ///   - temperature: Sampling temperature. Default 0.3.
    ///   - maxTokens: Maximum tokens per generation. Default 512.
    public func openSession(temperature: Float = 0.3, maxTokens: Int = 512) async throws {
        try ensureReady()
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            inferenceQueue.async { [self] in
                do {
                    if let s = chatSession {
                        litert_lm_session_delete(s)
                        chatSession = nil
                    }
                    if let c = chatSessionConfig {
                        litert_lm_session_config_delete(c)
                        chatSessionConfig = nil
                    }

                    guard let eng = engine else { throw LiteRTLMError.modelNotLoaded }
                    let (session, config) = try createSession(
                        engine: eng, temperature: temperature, maxTokens: Int32(maxTokens)
                    )
                    chatSession = session
                    chatSessionConfig = config
                    Self.log.info("Persistent session opened")
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Close the persistent session, freeing KV cache memory.
    public func closeSession() {
        inferenceQueue.async { [self] in
            guard chatSession != nil else { return }
            if let s = chatSession {
                logSessionBenchmark(s)
                litert_lm_session_delete(s)
                chatSession = nil
            }
            if let c = chatSessionConfig {
                litert_lm_session_config_delete(c)
                chatSessionConfig = nil
            }
            Self.log.info("Persistent session closed")
        }
    }

    // MARK: - Persistent Conversation (Multimodal KV Cache Reuse)
    //
    // Like the text-only persistent session above, but uses the Conversation
    // API — supporting images, audio, and text. The conversation's KV cache
    // persists across turns, so follow-up messages only prefill new tokens.

    private var multimodalConversation: OpaquePointer?
    private var multimodalConvConfig: OpaquePointer?
    private var multimodalSessionConfig: OpaquePointer?

    /// Set by `openConversation(...)` when `enableThinking: true`. Threaded as
    /// the `extra_context` JSON argument (`{"enable_thinking":true}`) on every
    /// subsequent send call. The Gemma 4 template auto-strips prior thoughts
    /// from KV context on the next turn, so there's no cache-reuse penalty.
    private var conversationThinkingEnabled: Bool = false

    /// Open a persistent multimodal conversation with KV cache reuse.
    ///
    /// Call once when a conversation begins. Subsequent calls to
    /// `conversationSend(...)` reuse this conversation's KV cache,
    /// reducing TTFT from ~20s to ~1-2s for follow-up turns.
    ///
    /// - Parameters:
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens per generation. Default 1024.
    public func openConversation(temperature: Float = 0.7, maxTokens: Int = 1024) async throws {
        try ensureReady()
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            inferenceQueue.async { [self] in
                do {
                    // Close existing conversation if any
                    if let c = multimodalConversation {
                        litert_lm_conversation_delete(c)
                        multimodalConversation = nil
                    }
                    if let c = multimodalConvConfig {
                        litert_lm_conversation_config_delete(c)
                        multimodalConvConfig = nil
                    }
                    if let c = multimodalSessionConfig {
                        litert_lm_session_config_delete(c)
                        multimodalSessionConfig = nil
                    }
                    conversationThinkingEnabled = false

                    guard let eng = engine else { throw LiteRTLMError.modelNotLoaded }

                    guard let sessionConfig = litert_lm_session_config_create() else {
                        throw LiteRTLMError.inferenceFailure("Failed to create session config")
                    }
                    litert_lm_session_config_set_max_output_tokens(sessionConfig, Int32(maxTokens))
                    var samplerParams = LiteRtLmSamplerParams(
                        type: kLiteRtLmSamplerTypeTopP, top_k: 40, top_p: 0.95,
                        temperature: temperature, seed: 0
                    )
                    litert_lm_session_config_set_sampler_params(sessionConfig, &samplerParams)

                    guard let convConfig = litert_lm_conversation_config_create() else {
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create conversation config")
                    }
                    litert_lm_conversation_config_set_session_config(convConfig, sessionConfig)

                    guard let conversation = litert_lm_conversation_create(eng, convConfig) else {
                        litert_lm_conversation_config_delete(convConfig)
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create conversation")
                    }

                    multimodalConversation = conversation
                    multimodalConvConfig = convConfig
                    multimodalSessionConfig = sessionConfig
                    Self.log.info("Persistent multimodal conversation opened")
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Send a message in the persistent multimodal conversation.
    ///
    /// Each call reuses the conversation's KV cache. Pass any combination of
    /// audio, images, and text — or just text for a follow-up question.
    ///
    /// - Parameters:
    ///   - audioData: Array of raw audio bytes. Pass empty array (default) for non-audio turns.
    ///   - audioFormat: Audio container format. Default `.wav`.
    ///   - imagesData: Array of raw image bytes. Pass empty array (default) for non-image turns.
    ///   - prompt: Text prompt for this turn.
    ///   - maxImageDimension: Resize image long edge to this value. Default 1024.
    /// - Returns: Generated text response.
    public func conversationSend(
        audioData: [Data] = [],
        audioFormat: AudioFormat = .wav,
        imagesData: [Data] = [],
        prompt: String,
        maxImageDimension: Int = 1024
    ) async throws -> String {
        try ensureReady()

        // Prepare media files
        var tempURLs: [URL] = []
        var audioPaths: [String] = []
        var imagePaths: [String] = []

        do {
            for (i, data) in audioData.enumerated() {
                guard !data.isEmpty else {
                    throw LiteRTLMError.inferenceFailure("Audio data \(i + 1) is empty")
                }
                let url = Self.makeTempURL(extension: audioFormat.rawValue)
                try data.write(to: url)
                tempURLs.append(url)
                audioPaths.append(url.path)
            }
            for (i, data) in imagesData.enumerated() {
                guard let jpegData = Self.prepareImageForVision(data, maxDimension: maxImageDimension) else {
                    throw LiteRTLMError.inferenceFailure("Failed to convert image \(i + 1) to JPEG")
                }
                let url = Self.makeTempURL(extension: "jpg")
                try jpegData.write(to: url)
                tempURLs.append(url)
                imagePaths.append(url.path)
            }
        } catch {
            Self.cleanupTempFiles(tempURLs)
            throw error
        }

        let messageJSON = Self.buildMultimodalMessageJSON(
            audioPaths: audioPaths, imagePaths: imagePaths, text: prompt
        )

        let urlsToCleanup = tempURLs
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, any Error>) in
            self.inferenceQueue.async { [self, urlsToCleanup] in
                defer { Self.cleanupTempFiles(urlsToCleanup) }
                do {
                    guard let conversation = self.multimodalConversation else {
                        throw LiteRTLMError.inferenceFailure(
                            "No persistent conversation open — call openConversation() first"
                        )
                    }

                    guard let response = messageJSON.withCString({ msgPtr in
                        litert_lm_conversation_send_message(conversation, msgPtr, nil)
                    }) else {
                        throw LiteRTLMError.inferenceFailure("Conversation returned no response")
                    }
                    defer { litert_lm_json_response_delete(response) }

                    guard let responsePtr = litert_lm_json_response_get_string(response) else {
                        throw LiteRTLMError.inferenceFailure("Response string is NULL")
                    }

                    let result = Self.extractTextFromConversationResponse(String(cString: responsePtr))
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Tool Calling (Mei patch)
    //
    // The C API on litert_lm_conversation_config_create already accepts
    // `system_message_json`, `tools_json`, and `enable_constrained_decoding`,
    // but the upstream wrapper passes nil for all three. The methods below
    // expose those parameters and return the full JSON response from the
    // model so callers can inspect `tool_calls`. They mirror the existing
    // `openConversation` / `conversationSend` shape.

    /// Open a persistent multimodal conversation with tool declarations.
    ///
    /// `toolsJSON` must be a JSON array of tool descriptors as documented in
    /// the LiteRT-LM tool-use guide:
    /// `[ { "name": "...", "description": "...", "parameters": { ... } }, ... ]`.
    /// `systemMessage` is optional context attached to the conversation.
    /// `enableConstrainedDecoding` turns on schema-guided decoding when the
    /// model supports it.
    public func openConversation(
        systemMessage: String?,
        toolsJSON: String?,
        temperature: Float = 0.7,
        maxTokens: Int = 1024,
        enableConstrainedDecoding: Bool = false,
        enableThinking: Bool = false
    ) async throws {
        try ensureReady()
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            inferenceQueue.async { [self] in
                do {
                    if let c = multimodalConversation {
                        litert_lm_conversation_delete(c)
                        multimodalConversation = nil
                    }
                    if let c = multimodalConvConfig {
                        litert_lm_conversation_config_delete(c)
                        multimodalConvConfig = nil
                    }
                    if let c = multimodalSessionConfig {
                        litert_lm_session_config_delete(c)
                        multimodalSessionConfig = nil
                    }
                    conversationThinkingEnabled = enableThinking

                    guard let eng = engine else { throw LiteRTLMError.modelNotLoaded }

                    guard let sessionConfig = litert_lm_session_config_create() else {
                        throw LiteRTLMError.inferenceFailure("Failed to create session config")
                    }
                    litert_lm_session_config_set_max_output_tokens(sessionConfig, Int32(maxTokens))
                    var samplerParams = LiteRtLmSamplerParams(
                        type: kLiteRtLmSamplerTypeTopP, top_k: 40, top_p: 0.95,
                        temperature: temperature, seed: 0
                    )
                    litert_lm_session_config_set_sampler_params(sessionConfig, &samplerParams)

                    let systemJSON: String? = systemMessage.flatMap { msg -> String? in
                        let payload: [String: Any] = ["role": "system", "content": msg]
                        guard let data = try? JSONSerialization.data(withJSONObject: payload),
                              let s = String(data: data, encoding: .utf8) else { return nil }
                        return s
                    }

                    guard let convConfig = litert_lm_conversation_config_create() else {
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create conversation config")
                    }
                    litert_lm_conversation_config_set_session_config(convConfig, sessionConfig)
                    if let sysJSON = systemJSON {
                        litert_lm_conversation_config_set_system_message(convConfig, sysJSON)
                    }
                    if let tools = toolsJSON {
                        litert_lm_conversation_config_set_tools(convConfig, tools)
                    }
                    litert_lm_conversation_config_set_enable_constrained_decoding(convConfig, enableConstrainedDecoding)

                    guard let conversation = litert_lm_conversation_create(eng, convConfig) else {
                        litert_lm_conversation_config_delete(convConfig)
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create conversation")
                    }

                    multimodalConversation = conversation
                    multimodalConvConfig = convConfig
                    multimodalSessionConfig = sessionConfig
                    Self.log.info("Persistent conversation opened with tools")
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Send a text message in the persistent conversation and return the raw
    /// JSON the model produced. Use this when you need to inspect
    /// `tool_calls` rather than just the text reply.
    public func conversationSendRaw(prompt: String) async throws -> String {
        try ensureReady()
        let messageJSON = Self.buildMultimodalMessageJSON(
            audioPaths: [], imagePaths: [], text: prompt
        )
        return try await sendRawMessage(messageJSON: messageJSON)
    }

    /// Multimodal counterpart to `conversationSendRaw`. Sends images and/or
    /// audio alongside the text prompt within an open tool-call conversation
    /// and returns the raw JSON reply so callers can inspect `tool_calls`.
    /// Mirrors `conversationSend(audioData:imagesData:prompt:)` but routes
    /// through `sendRawMessage` instead of swallowing the response.
    public func conversationSendMultimodalRaw(
        imagesData: [Data] = [],
        audioData: [Data] = [],
        audioFormat: AudioFormat = .wav,
        prompt: String,
        maxImageDimension: Int = 1024
    ) async throws -> String {
        try ensureReady()

        var tempURLs: [URL] = []
        var audioPaths: [String] = []
        var imagePaths: [String] = []

        do {
            for (i, data) in audioData.enumerated() {
                guard !data.isEmpty else {
                    throw LiteRTLMError.inferenceFailure("Audio data \(i + 1) is empty")
                }
                let url = Self.makeTempURL(extension: audioFormat.rawValue)
                try data.write(to: url)
                tempURLs.append(url)
                audioPaths.append(url.path)
            }
            for (i, data) in imagesData.enumerated() {
                guard let jpegData = Self.prepareImageForVision(data, maxDimension: maxImageDimension) else {
                    throw LiteRTLMError.inferenceFailure("Failed to convert image \(i + 1) to JPEG")
                }
                let url = Self.makeTempURL(extension: "jpg")
                try jpegData.write(to: url)
                tempURLs.append(url)
                imagePaths.append(url.path)
            }
        } catch {
            Self.cleanupTempFiles(tempURLs)
            throw error
        }

        let messageJSON = Self.buildMultimodalMessageJSON(
            audioPaths: audioPaths, imagePaths: imagePaths, text: prompt
        )

        defer { Self.cleanupTempFiles(tempURLs) }
        return try await sendRawMessage(messageJSON: messageJSON)
    }

    /// A prior turn replayed into the conversation KV cache on the first
    /// send of a reopened session. The C API accepts a JSON array of
    /// messages, so we prepend these as real `role: user` / `role:
    /// assistant` entries before the new user turn — far more reliable
    /// than stuffing the same text into the system prompt and hoping the
    /// model treats it as history.
    public struct PriorTurn: Sendable {
        public enum Role: String, Sendable { case user, assistant }
        public let role: Role
        public let text: String
        public init(role: Role, text: String) {
            self.role = role
            self.text = text
        }
    }

    /// Send the new user message with prior turns prefilled into the
    /// conversation as proper role-tagged messages. Use on the first
    /// send after `openConversation` when reopening a conversation whose
    /// prior turns aren't yet in the engine's KV cache. Subsequent turns
    /// in the same session should keep using `conversationSendRaw` — the
    /// cache already carries them.
    ///
    /// Passing an empty `priorTurns` array is equivalent to calling
    /// `conversationSendRaw(prompt:)` directly.
    public func conversationSendWithHistory(
        priorTurns: [PriorTurn],
        newUserMessage: String
    ) async throws -> String {
        if priorTurns.isEmpty {
            return try await conversationSendRaw(prompt: newUserMessage)
        }
        var payload: [[String: Any]] = priorTurns.map { turn in
            [
                "role": turn.role.rawValue,
                "content": turn.text
            ]
        }
        payload.append([
            "role": "user",
            "content": [["type": "text", "text": newUserMessage]]
        ])
        let messageJSON = (try? JSONSerialization.data(withJSONObject: payload))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
        return try await sendRawMessage(messageJSON: messageJSON)
    }

    /// Shape of the `role: "tool"` message to emit to LiteRT-LM's Conversation
    /// API. Gemma 4 E2B (LiteRT-LM v0.10.2) silently drops the default
    /// `.contentDictWithToolName` shape — the C++ template lowers the
    /// payload but the model never sees the values — so callers need a way
    /// to experiment with alternate shapes until upstream lands a fix.
    public enum ToolResultPayloadShape: Sendable {
        /// `{role:"tool", content:{tool_name, ...payload}}` — matches the
        /// LiteRT-LM tool-use doc. Default for backwards compatibility.
        case contentDictWithToolName
        /// `{role:"tool", name:<toolName>, content:<payload-dict>}`.
        /// Tried in mei.5, reverted because the C template rendered
        /// `response:unknown{...}`.
        case nameAndContentDict
        /// `{role:"tool", name:<toolName>, content:"<stringified-payload>"}`
        /// — OpenAI-classic, flat-string content.
        case nameAndContentString
        /// `{role:"tool", content:[{type:"text", text:"<stringified-payload>"}]}`
        /// — mirrors the shape the user-role path uses. Hypothesis: the C
        /// template's content walker only renders typed-part arrays.
        case contentArrayTyped
        /// `{role:"tool", name:<toolName>, content:[{type:"text", text:"..."}]}`
        /// — union of `.nameAndContentString` and `.contentArrayTyped`.
        case nameAndContentArrayTyped
    }

    /// Send tool execution results back to the model in the persistent
    /// conversation. `results` are tool-name → JSON-serializable payload
    /// pairs; each is sent as a separate `role: "tool"` message in a single
    /// batch so the model can fold them into one follow-up turn.
    /// Returns the model's raw JSON reply (which may be a final text answer
    /// or another `tool_calls` round).
    ///
    /// `shape` selects the `role: "tool"` message shape. Defaults to the
    /// LiteRT-LM tool-use doc shape. See `ToolResultPayloadShape` for
    /// alternatives to try when the default is silently dropped.
    /// Ref: https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/api/cpp/tool-use.md
    public func sendToolResults(
        _ results: [(toolName: String, payload: [String: Any])],
        shape: ToolResultPayloadShape = .contentDictWithToolName
    ) async throws -> String {
        try ensureReady()
        let messages: [[String: Any]] = results.map { result in
            Self.buildToolResultMessage(toolName: result.toolName, payload: result.payload, shape: shape)
        }

        let messageJSON: String
        if messages.count == 1 {
            messageJSON = (try? JSONSerialization.data(withJSONObject: messages[0]))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        } else {
            // Multiple tool results — send as JSON array; LiteRT-LM accepts
            // either a single message object or an array of messages.
            messageJSON = (try? JSONSerialization.data(withJSONObject: messages))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
        }
        return try await sendRawMessage(messageJSON: messageJSON)
    }

    private nonisolated static func buildToolResultMessage(
        toolName: String,
        payload: [String: Any],
        shape: ToolResultPayloadShape
    ) -> [String: Any] {
        switch shape {
        case .contentDictWithToolName:
            var content = payload
            content["tool_name"] = toolName
            return ["role": "tool", "content": content]
        case .nameAndContentDict:
            return ["role": "tool", "name": toolName, "content": payload]
        case .nameAndContentString:
            return ["role": "tool", "name": toolName, "content": stringifyToolPayload(payload)]
        case .contentArrayTyped:
            return [
                "role": "tool",
                "content": [["type": "text", "text": stringifyToolPayload(payload)]]
            ]
        case .nameAndContentArrayTyped:
            return [
                "role": "tool",
                "name": toolName,
                "content": [["type": "text", "text": stringifyToolPayload(payload)]]
            ]
        }
    }

    /// Collapse a tool payload dict to a readable string. When the dict has a
    /// single natural-text field (`output`/`result`/`text`/`content`) we inline
    /// the value so the model doesn't have to parse JSON; otherwise we fall
    /// back to a JSON encoding.
    private nonisolated static func stringifyToolPayload(_ payload: [String: Any]) -> String {
        let textKeys: Set<String> = ["output", "result", "text", "content"]
        if payload.count == 1, let key = payload.keys.first, textKeys.contains(key),
           let text = payload[key] as? String {
            return text
        }
        if let data = try? JSONSerialization.data(withJSONObject: payload),
           let s = String(data: data, encoding: .utf8) {
            return s
        }
        return String(describing: payload)
    }

    /// Parse `tool_calls` out of a raw conversation JSON response.
    /// Returns an empty array when the model produced a plain text reply.
    /// String values are scrubbed of Gemma-4 string-delimiter sentinels
    /// (`<|"|>`, `<|'|>`) the detokenizer sometimes leaks into argument
    /// bodies.
    public nonisolated static func parseToolCalls(from rawJSON: String) -> [ParsedToolCall] {
        guard let data = rawJSON.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return []
        }
        guard let calls = obj["tool_calls"] as? [[String: Any]] else { return [] }
        return calls.compactMap { call in
            guard let function = call["function"] as? [String: Any],
                  let name = function["name"] as? String else { return nil }
            let arguments = (function["arguments"] as? [String: Any]) ?? [:]
            let mapped = arguments.reduce(into: [String: ParsedToolArgument]()) { acc, pair in
                acc[pair.key] = ParsedToolArgument(any: pair.value)
            }
            return ParsedToolCall(name: stripSentinels(name), arguments: mapped)
        }
    }

    /// Strip Gemma-4 string-delimiter sentinels (`<|"|>`, `<|'|>`) that
    /// sometimes leak into decoded tool-call string values. Callers can run
    /// this over any raw model text to get the intended value.
    public nonisolated static func stripSentinels(_ raw: String) -> String {
        var out = raw
        for token in ["<|\"|>", "<|'|>"] {
            out = out.replacingOccurrences(of: token, with: "")
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// JSON-typed argument carried alongside a parsed tool call.
    public enum ParsedToolArgument: Sendable, Equatable {
        case string(String)
        case stringArray([String])
        case bool(Bool)
        case number(Double)
        case null

        public var stringValue: String? {
            if case .string(let s) = self { return s }
            return nil
        }

        public var stringArrayValue: [String]? {
            if case .stringArray(let xs) = self { return xs }
            return nil
        }

        init(any: Any) {
            if any is NSNull { self = .null; return }
            if let s = any as? String {
                self = .string(LiteRTLMEngine.stripSentinels(s))
                return
            }
            if let xs = any as? [String] {
                self = .stringArray(xs.map(LiteRTLMEngine.stripSentinels))
                return
            }
            if let xs = any as? [Any] {
                self = .stringArray(xs.map { LiteRTLMEngine.stripSentinels(String(describing: $0)) })
                return
            }
            if let b = any as? Bool { self = .bool(b); return }
            if let n = any as? NSNumber { self = .number(n.doubleValue); return }
            self = .string(LiteRTLMEngine.stripSentinels(String(describing: any)))
        }
    }

    /// Lightweight DTO for a parsed tool call from `tool_calls`.
    public struct ParsedToolCall: Sendable, Equatable {
        public let name: String
        public let arguments: [String: ParsedToolArgument]
    }

    // MARK: - Typed Tool Calling (Gemma 4)
    //
    // Convenience layer over the raw-JSON tool-use methods above. Callers work
    // with Swift structs instead of hand-rolling the OpenAI-shape tools array
    // and re-parsing the response JSON on every turn.

    /// Open a persistent conversation with typed tool declarations and optional
    /// thinking mode.
    ///
    /// - Parameters:
    ///   - systemPrompt: Optional system message prefixed to the conversation.
    ///   - tools: Tool declarations. Empty array means "no tools."
    ///   - enableConstrainedDecoding: If `nil` (default), auto-enabled when
    ///     `tools` is non-empty. Pass `false` explicitly to disable grammar
    ///     constraints even with tools declared. Never auto-enabled when no
    ///     tools are present — library versions have been observed to hang
    ///     when constrained decoding is on but no tools are declared.
    ///   - enableThinking: When `true`, every subsequent send emits thought
    ///     tokens before the final answer. Requires a Gemma 4 model.
    public func openConversation(
        systemPrompt: String? = nil,
        tools: [LiteRTLMTool] = [],
        enableConstrainedDecoding: Bool? = nil,
        enableThinking: Bool = false,
        temperature: Float = 0.7,
        maxTokens: Int = 1024
    ) async throws {
        let toolsJSON: String? = tools.isEmpty ? nil : try buildToolsJSON(tools)
        let constrain = enableConstrainedDecoding ?? !tools.isEmpty
        try await openConversation(
            systemMessage: systemPrompt,
            toolsJSON: toolsJSON,
            temperature: temperature,
            maxTokens: maxTokens,
            enableConstrainedDecoding: constrain,
            enableThinking: enableThinking
        )
    }

    /// Send a user turn and receive a typed `LiteRTLMTurn`. If the model chose
    /// to call tools, the `.toolCalls` case carries the parsed invocations;
    /// otherwise `.text` carries the final answer.
    public func conversationSendTurn(prompt: String) async throws -> LiteRTLMTurn {
        let rawJSON = try await conversationSendRaw(prompt: prompt)
        return Self.parseTurn(rawJSON: rawJSON)
    }

    /// Send tool execution results and receive the model's typed follow-up
    /// turn. The follow-up may be another round of tool calls or a final
    /// text answer — inspect the returned `LiteRTLMTurn`.
    public func sendToolResultsTurn(
        _ results: [(toolName: String, payload: [String: Any])]
    ) async throws -> LiteRTLMTurn {
        let rawJSON = try await sendToolResults(results)
        return Self.parseTurn(rawJSON: rawJSON)
    }

    /// Parse a raw conversation JSON response into a typed turn. Returns
    /// `.toolCalls` when the payload carries a `tool_calls` array, otherwise
    /// `.text` with the extracted text content.
    nonisolated static func parseTurn(rawJSON: String) -> LiteRTLMTurn {
        let toolCalls = parseToolCalls(from: rawJSON)
        if !toolCalls.isEmpty { return .toolCalls(toolCalls) }
        return .text(extractTextFromConversationResponse(rawJSON))
    }

    private func sendRawMessage(
        messageJSON: String,
        extraContextJSON: String? = nil
    ) async throws -> String {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, any Error>) in
            self.inferenceQueue.async { [self] in
                do {
                    guard let conversation = self.multimodalConversation else {
                        throw LiteRTLMError.noConversationOpen
                    }

                    // Caller-supplied context wins; otherwise fall back to the
                    // thinking flag captured at openConversation time.
                    let effectiveContext = extraContextJSON
                        ?? (self.conversationThinkingEnabled ? "{\"enable_thinking\":true}" : nil)

                    guard let response = messageJSON.withCString({ msgPtr -> OpaquePointer? in
                        withOptionalCString(effectiveContext) { ctxPtr in
                            litert_lm_conversation_send_message(conversation, msgPtr, ctxPtr)
                        }
                    }) else {
                        throw LiteRTLMError.inferenceFailure("Conversation returned no response")
                    }
                    defer { litert_lm_json_response_delete(response) }

                    guard let responsePtr = litert_lm_json_response_get_string(response) else {
                        throw LiteRTLMError.inferenceFailure("Response string is NULL")
                    }
                    continuation.resume(returning: String(cString: responsePtr))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func withOptionalCString<R>(
        _ string: String?,
        _ body: (UnsafePointer<CChar>?) -> R
    ) -> R {
        if let string {
            return string.withCString { body($0) }
        } else {
            return body(nil)
        }
    }

    /// Stream a user turn as typed events (`.text`, `.thought`, `.toolCalls`).
    ///
    /// The underlying C API delivers plain-text chunks. We attempt to parse
    /// each chunk as JSON first — that's how tool-call chunks and the Gemma 4
    /// `channels.thought` payload arrive when thinking mode is on. Non-JSON
    /// chunks are surfaced verbatim as `.text`. The full accumulated output is
    /// re-parsed at stream-end in case the model emitted one final `tool_calls`
    /// JSON object split across chunks.
    ///
    /// Cancel in flight via `litert_lm_conversation_cancel_process` by calling
    /// `cancelConversation()`.
    public func conversationSendTurnStreaming(
        prompt: String
    ) -> AsyncThrowingStream<LiteRTLMStreamEvent, Error> {
        let messageJSON = Self.buildMultimodalMessageJSON(
            audioPaths: [], imagePaths: [], text: prompt
        )
        return sendRawMessageStreaming(messageJSON: messageJSON)
    }

    /// Streaming counterpart to `conversationSendWithHistory`. Replays prior
    /// turns into the open conversation as proper role-tagged messages and
    /// streams the model's reply as typed events. Use on the first send
    /// after `openConversation` when reopening a conversation whose prior
    /// turns aren't yet in the engine's KV cache.
    public func conversationSendWithHistoryStreaming(
        priorTurns: [PriorTurn],
        newUserMessage: String
    ) -> AsyncThrowingStream<LiteRTLMStreamEvent, Error> {
        if priorTurns.isEmpty {
            return conversationSendTurnStreaming(prompt: newUserMessage)
        }
        var payload: [[String: Any]] = priorTurns.map { turn in
            [
                "role": turn.role.rawValue,
                "content": turn.text
            ]
        }
        payload.append([
            "role": "user",
            "content": [["type": "text", "text": newUserMessage]]
        ])
        let messageJSON = (try? JSONSerialization.data(withJSONObject: payload))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
        return sendRawMessageStreaming(messageJSON: messageJSON)
    }

    /// Streaming counterpart to `sendToolResults`. Submits one or more
    /// `role: "tool"` messages and streams the model's follow-up reply
    /// (which may itself be another `tool_calls` round). Same `shape`
    /// semantics as `sendToolResults`.
    public func sendToolResultsStreaming(
        _ results: [(toolName: String, payload: [String: Any])],
        shape: ToolResultPayloadShape = .contentDictWithToolName
    ) -> AsyncThrowingStream<LiteRTLMStreamEvent, Error> {
        let messages: [[String: Any]] = results.map { result in
            Self.buildToolResultMessage(toolName: result.toolName, payload: result.payload, shape: shape)
        }
        let messageJSON: String
        if messages.count == 1 {
            messageJSON = (try? JSONSerialization.data(withJSONObject: messages[0]))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        } else {
            messageJSON = (try? JSONSerialization.data(withJSONObject: messages))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
        }
        return sendRawMessageStreaming(messageJSON: messageJSON)
    }

    /// Shared streaming primitive. Owns the C callback bridge so the public
    /// streaming entry points stay focused on building their message JSON.
    private func sendRawMessageStreaming(
        messageJSON: String
    ) -> AsyncThrowingStream<LiteRTLMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            self.inferenceQueue.async { [self] in
                guard let conversation = self.multimodalConversation else {
                    continuation.finish(throwing: LiteRTLMError.noConversationOpen)
                    return
                }

                let extraContext: String? = self.conversationThinkingEnabled
                    ? "{\"enable_thinking\":true}" : nil

                let streamDone = DispatchSemaphore(value: 0)
                let state = ConversationStreamState(continuation: continuation, doneSemaphore: streamDone)
                let statePtr = Unmanaged.passRetained(state).toOpaque()

                let result = messageJSON.withCString { msgPtr -> Int32 in
                    withOptionalCString(extraContext) { ctxPtr in
                        litert_lm_conversation_send_message_stream(
                            conversation, msgPtr, ctxPtr,
                            { callbackData, chunk, isFinal, errorMsg in
                                guard let cbData = callbackData else { return }
                                let st = Unmanaged<ConversationStreamState>.fromOpaque(cbData)
                                    .takeUnretainedValue()

                                let errorMessage: String? = {
                                    guard let errorMsg else { return nil }
                                    let msg = String(cString: errorMsg)
                                    return msg.isEmpty ? nil : msg
                                }()

                                if let chunk, errorMessage == nil {
                                    let text = String(cString: chunk)
                                    if !text.isEmpty {
                                        st.buffer.append(text)
                                        for event in LiteRTLMEngine.streamEvents(fromChunk: text) {
                                            if case .toolCalls = event { st.yieldedToolCalls = true }
                                            st.continuation.yield(event)
                                        }
                                    }
                                }

                                if isFinal || errorMessage != nil {
                                    if let error = errorMessage {
                                        st.continuation.finish(throwing: LiteRTLMError.inferenceFailure(error))
                                    } else {
                                        // Final pass: if the whole payload parses as a
                                        // tool_calls envelope, surface them. This covers
                                        // libraries that stream the tool-call JSON across
                                        // chunks without per-chunk parseability.
                                        let full = st.buffer
                                        let toolCalls = LiteRTLMEngine.parseToolCalls(from: full)
                                        if !toolCalls.isEmpty && !st.yieldedToolCalls {
                                            st.continuation.yield(.toolCalls(toolCalls))
                                        }
                                        st.continuation.finish()
                                    }
                                    let semaphore = st.doneSemaphore
                                    Unmanaged<ConversationStreamState>.fromOpaque(cbData).release()
                                    semaphore.signal()
                                }
                            },
                            statePtr
                        )
                    }
                }

                if result != 0 {
                    Unmanaged<ConversationStreamState>.fromOpaque(statePtr).release()
                    continuation.finish(throwing: LiteRTLMError.inferenceFailure("Failed to start conversation stream"))
                    return
                }

                streamDone.wait()
            }
        }
    }

    /// Cancel an in-flight streaming or blocking conversation send. Safe to
    /// call when no send is in flight.
    public func cancelConversation() {
        inferenceQueue.async { [self] in
            if let conversation = multimodalConversation {
                litert_lm_conversation_cancel_process(conversation)
            }
        }
    }

    /// Per-chunk event parser used by the streaming path. Tries JSON first
    /// (to surface structured `channels.thought` / `tool_calls` chunks),
    /// falls back to emitting a single `.text` event.
    nonisolated static func streamEvents(fromChunk chunk: String) -> [LiteRTLMStreamEvent] {
        guard let data = chunk.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return [.text(chunk)]
        }

        var events: [LiteRTLMStreamEvent] = []

        if let channels = obj["channels"] as? [String: Any],
           let thought = channels["thought"] as? String, !thought.isEmpty {
            events.append(.thought(thought))
        }
        if let thought = obj["thought"] as? String, !thought.isEmpty {
            events.append(.thought(thought))
        }

        let calls = parseToolCalls(from: chunk)
        if !calls.isEmpty {
            events.append(.toolCalls(calls))
        }

        if let content = obj["content"] as? [[String: Any]] {
            for part in content {
                if let text = part["text"] as? String, !text.isEmpty {
                    events.append(.text(text))
                }
            }
        } else if let text = obj["text"] as? String, !text.isEmpty {
            events.append(.text(text))
        }

        return events.isEmpty ? [.text(chunk)] : events
    }

    /// Close the persistent multimodal conversation, freeing KV cache memory.
    public func closeConversation() {
        inferenceQueue.async { [self] in
            guard multimodalConversation != nil else { return }
            if let c = multimodalConversation {
                litert_lm_conversation_delete(c)
                multimodalConversation = nil
            }
            if let c = multimodalConvConfig {
                litert_lm_conversation_config_delete(c)
                multimodalConvConfig = nil
            }
            if let c = multimodalSessionConfig {
                litert_lm_session_config_delete(c)
                multimodalSessionConfig = nil
            }
            conversationThinkingEnabled = false
            Self.log.info("Persistent multimodal conversation closed")
        }
    }

    /// Stream text using the persistent session.
    ///
    /// `input` should be ONLY the new turn content — the session's KV cache
    /// already holds all previous context.
    ///
    /// - Parameter input: New input text for this turn.
    /// - Returns: An `AsyncThrowingStream` yielding text chunks.
    public func sessionGenerateStreaming(input: String) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            self.inferenceQueue.async { [self] in
                guard let session = self.chatSession else {
                    continuation.finish(throwing: LiteRTLMError.inferenceFailure("No persistent session open — call openSession() first"))
                    return
                }

                let streamDone = DispatchSemaphore(value: 0)
                let state = StreamCallbackState(continuation: continuation, doneSemaphore: streamDone)
                let statePtr = Unmanaged.passRetained(state).toOpaque()

                let result = input.withCString { textPtr -> Int32 in
                    var inputData = LiteRtLmInputData(
                        type: kLiteRtLmInputDataTypeText,
                        data: UnsafeRawPointer(textPtr),
                        size: strlen(textPtr)
                    )
                    return litert_lm_session_generate_content_stream(
                        session, &inputData, 1,
                        { callbackData, chunk, isFinal, errorMsg in
                            guard let cbData = callbackData else { return }
                            let st = Unmanaged<StreamCallbackState>.fromOpaque(cbData)
                                .takeUnretainedValue()

                            let errorMessage: String? = {
                                guard let errorMsg else { return nil }
                                let msg = String(cString: errorMsg)
                                return msg.isEmpty ? nil : msg
                            }()

                            if let chunk, errorMessage == nil {
                                let text = String(cString: chunk)
                                if !text.isEmpty { st.continuation.yield(text) }
                            }

                            if isFinal || errorMessage != nil {
                                if let error = errorMessage {
                                    st.continuation.finish(throwing: LiteRTLMError.inferenceFailure(error))
                                } else {
                                    st.continuation.finish()
                                }
                                let semaphore = st.doneSemaphore
                                Unmanaged<StreamCallbackState>.fromOpaque(cbData).release()
                                semaphore.signal()
                            }
                        },
                        statePtr
                    )
                }

                if result != 0 {
                    Unmanaged<StreamCallbackState>.fromOpaque(statePtr).release()
                    continuation.finish(throwing: LiteRTLMError.inferenceFailure("Failed to start stream"))
                    return
                }

                streamDone.wait()
                self.logSessionBenchmark(session)
            }
        }
    }

    // MARK: - Private: Session-based Inference

    private func runSessionInference(
        prompt: String,
        temperature: Float,
        maxTokens: Int32
    ) async throws -> String {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, any Error>) in
            self.inferenceQueue.async { [self] in
                do {
                    guard let eng = self.engine else { throw LiteRTLMError.modelNotLoaded }

                    let (session, sessionConfig) = try self.createSession(
                        engine: eng, temperature: temperature, maxTokens: maxTokens
                    )
                    defer {
                        litert_lm_session_delete(session)
                        litert_lm_session_config_delete(sessionConfig)
                    }

                    let output = prompt.withCString { textPtr -> String? in
                        var input = LiteRtLmInputData(
                            type: kLiteRtLmInputDataTypeText,
                            data: UnsafeRawPointer(textPtr),
                            size: strlen(textPtr)
                        )
                        guard let responses = litert_lm_session_generate_content(session, &input, 1) else {
                            return nil
                        }
                        defer { litert_lm_responses_delete(responses) }
                        return self.extractResponseText(responses)
                    }

                    guard let result = output else {
                        throw LiteRTLMError.inferenceFailure("generate_content returned no output")
                    }

                    self.logSessionBenchmark(session)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func runSessionInferenceStreaming(
        prompt: String,
        temperature: Float,
        maxTokens: Int32
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            self.inferenceQueue.async { [self] in
                do {
                    try self.ensureReady()
                    guard let eng = self.engine else {
                        continuation.finish(throwing: LiteRTLMError.modelNotLoaded)
                        return
                    }

                    let (session, sessionConfig) = try self.createSession(
                        engine: eng, temperature: temperature, maxTokens: maxTokens
                    )

                    let streamDone = DispatchSemaphore(value: 0)
                    let state = StreamCallbackState(continuation: continuation, doneSemaphore: streamDone)
                    let statePtr = Unmanaged.passRetained(state).toOpaque()

                    let result = prompt.withCString { textPtr -> Int32 in
                        var input = LiteRtLmInputData(
                            type: kLiteRtLmInputDataTypeText,
                            data: UnsafeRawPointer(textPtr),
                            size: strlen(textPtr)
                        )
                        return litert_lm_session_generate_content_stream(
                            session, &input, 1,
                            { callbackData, chunk, isFinal, errorMsg in
                                guard let cbData = callbackData else { return }
                                let st = Unmanaged<StreamCallbackState>.fromOpaque(cbData)
                                    .takeUnretainedValue()

                                let errorMessage: String? = {
                                    guard let errorMsg else { return nil }
                                    let msg = String(cString: errorMsg)
                                    return msg.isEmpty ? nil : msg
                                }()

                                if let chunk, errorMessage == nil {
                                    let text = String(cString: chunk)
                                    if !text.isEmpty { st.continuation.yield(text) }
                                }

                                if isFinal || errorMessage != nil {
                                    if let error = errorMessage {
                                        st.continuation.finish(throwing: LiteRTLMError.inferenceFailure(error))
                                    } else {
                                        st.continuation.finish()
                                    }
                                    let semaphore = st.doneSemaphore
                                    Unmanaged<StreamCallbackState>.fromOpaque(cbData).release()
                                    semaphore.signal()
                                }
                            },
                            statePtr
                        )
                    }

                    if result != 0 {
                        Unmanaged<StreamCallbackState>.fromOpaque(statePtr).release()
                        litert_lm_session_delete(session)
                        litert_lm_session_config_delete(sessionConfig)
                        continuation.finish(throwing: LiteRTLMError.inferenceFailure("Failed to start stream"))
                        return
                    }

                    streamDone.wait()
                    self.logSessionBenchmark(session)
                    litert_lm_session_delete(session)
                    litert_lm_session_config_delete(sessionConfig)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Private Helpers

    private func ensureReady() throws {
        guard status == .ready else { throw LiteRTLMError.modelNotLoaded }
    }

    private func createSession(
        engine eng: OpaquePointer,
        temperature: Float,
        maxTokens: Int32
    ) throws -> (session: OpaquePointer, config: OpaquePointer) {
        guard let sessionConfig = litert_lm_session_config_create() else {
            throw LiteRTLMError.inferenceFailure("Failed to create session config")
        }

        litert_lm_session_config_set_max_output_tokens(sessionConfig, maxTokens)
        var samplerParams = LiteRtLmSamplerParams(
            type: kLiteRtLmSamplerTypeTopP, top_k: 40, top_p: 0.95,
            temperature: temperature, seed: 0
        )
        litert_lm_session_config_set_sampler_params(sessionConfig, &samplerParams)

        guard let session = litert_lm_engine_create_session(eng, sessionConfig) else {
            litert_lm_session_config_delete(sessionConfig)
            throw LiteRTLMError.inferenceFailure("Failed to create session")
        }

        return (session, sessionConfig)
    }

    private func extractResponseText(_ responses: OpaquePointer) -> String? {
        let numCandidates = litert_lm_responses_get_num_candidates(responses)
        guard numCandidates > 0,
              let resultPtr = litert_lm_responses_get_response_text_at(responses, 0) else {
            return nil
        }
        return String(cString: resultPtr)
    }

    private func logSessionBenchmark(_ session: OpaquePointer) {
        guard let info = litert_lm_session_get_benchmark_info(session) else { return }
        defer { litert_lm_benchmark_info_delete(info) }

        let initTime = litert_lm_benchmark_info_get_total_init_time_in_second(info)
        let ttft = litert_lm_benchmark_info_get_time_to_first_token(info)
        let numDecode = litert_lm_benchmark_info_get_num_decode_turns(info)
        let numPrefill = litert_lm_benchmark_info_get_num_prefill_turns(info)

        Self.log.info("Benchmark: init=\(String(format: "%.2f", initTime))s, TTFT=\(String(format: "%.2f", ttft))s")

        for i in 0..<numPrefill {
            let tps = litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(info, Int32(i))
            let count = litert_lm_benchmark_info_get_prefill_token_count_at(info, Int32(i))
            Self.log.info("  Prefill[\(i)]: \(count) tokens @ \(String(format: "%.1f", tps)) tok/s")
        }
        for i in 0..<numDecode {
            let tps = litert_lm_benchmark_info_get_decode_tokens_per_sec_at(info, Int32(i))
            let count = litert_lm_benchmark_info_get_decode_token_count_at(info, Int32(i))
            Self.log.info("  Decode[\(i)]: \(count) tokens @ \(String(format: "%.1f", tps)) tok/s")
        }
    }

    // MARK: - Private: Conversation-based Inference (Vision / Audio / Multimodal)

    /// Shared helper for all Conversation API calls (vision, audio, multimodal).
    /// Handles session/conversation lifecycle and temp file cleanup.
    private func runConversationInference(
        messageJSON: String,
        tempURLs: [URL],
        temperature: Float,
        maxTokens: Int
    ) async throws -> String {
        let urlsToCleanup = tempURLs
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, any Error>) in
            self.inferenceQueue.async { [self, urlsToCleanup] in
                defer {
                    for url in urlsToCleanup {
                        try? FileManager.default.removeItem(at: url)
                    }
                }
                do {
                    guard let eng = self.engine else { throw LiteRTLMError.modelNotLoaded }

                    guard let sessionConfig = litert_lm_session_config_create() else {
                        throw LiteRTLMError.inferenceFailure("Failed to create session config")
                    }
                    litert_lm_session_config_set_max_output_tokens(sessionConfig, Int32(maxTokens))
                    var samplerParams = LiteRtLmSamplerParams(
                        type: kLiteRtLmSamplerTypeTopP, top_k: 40, top_p: 0.95,
                        temperature: temperature, seed: 0
                    )
                    litert_lm_session_config_set_sampler_params(sessionConfig, &samplerParams)

                    guard let convConfig = litert_lm_conversation_config_create() else {
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create conversation config")
                    }
                    litert_lm_conversation_config_set_session_config(convConfig, sessionConfig)

                    guard let conversation = litert_lm_conversation_create(eng, convConfig) else {
                        litert_lm_conversation_config_delete(convConfig)
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create conversation")
                    }
                    defer {
                        litert_lm_conversation_delete(conversation)
                        litert_lm_conversation_config_delete(convConfig)
                        litert_lm_session_config_delete(sessionConfig)
                    }

                    guard let response = messageJSON.withCString({ msgPtr in
                        litert_lm_conversation_send_message(conversation, msgPtr, nil)
                    }) else {
                        throw LiteRTLMError.inferenceFailure("Conversation returned no response")
                    }
                    defer { litert_lm_json_response_delete(response) }

                    guard let responsePtr = litert_lm_json_response_get_string(response) else {
                        throw LiteRTLMError.inferenceFailure("Response string is NULL")
                    }

                    let result = Self.extractTextFromConversationResponse(String(cString: responsePtr))
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Media Helpers

    /// Create a uniquely-named temp file URL.
    nonisolated static func makeTempURL(extension ext: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + "." + ext)
    }

    /// Remove temp files, ignoring errors (best-effort cleanup).
    nonisolated static func cleanupTempFiles(_ urls: [URL]) {
        for url in urls {
            try? FileManager.default.removeItem(at: url)
        }
    }

    /// Convert any image format to JPEG and resize for vision inference.
    nonisolated static func prepareImageForVision(_ data: Data, maxDimension: Int = 1024) -> Data? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }

        let width = cgImage.width
        let height = cgImage.height

        let maxDim = maxDimension
        let scale: Double
        if width > height {
            scale = width > maxDim ? Double(maxDim) / Double(width) : 1.0
        } else {
            scale = height > maxDim ? Double(maxDim) / Double(height) : 1.0
        }

        let targetWidth = Int(Double(width) * scale)
        let targetHeight = Int(Double(height) * scale)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: targetWidth * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }

        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))

        guard let resizedImage = context.makeImage() else { return nil }

        let mutableData = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            mutableData, "public.jpeg" as CFString, 1, nil
        ) else { return nil }

        let options: [CFString: Any] = [kCGImageDestinationLossyCompressionQuality: 0.85]
        CGImageDestinationAddImage(destination, resizedImage, options as CFDictionary)

        guard CGImageDestinationFinalize(destination) else { return nil }
        return mutableData as Data
    }

    /// Build a Conversation API JSON message with any combination of audio, images, and text.
    nonisolated static func buildMultimodalMessageJSON(
        audioPaths: [String],
        imagePaths: [String],
        text: String
    ) -> String {
        var contentItems: [[String: Any]] = []
        for path in audioPaths {
            contentItems.append(["type": "audio", "path": path])
        }
        for path in imagePaths {
            contentItems.append(["type": "image", "path": path])
        }
        contentItems.append(["type": "text", "text": text])
        let message: [String: Any] = ["role": "user", "content": contentItems]
        guard let jsonData = try? JSONSerialization.data(withJSONObject: message),
              let jsonString = String(data: jsonData, encoding: .utf8) else {
            // Fallback: text-only, properly escaped via JSONSerialization
            let fallback: [String: Any] = ["role": "user", "content": [["type": "text", "text": text]]]
            let fallbackData = (try? JSONSerialization.data(withJSONObject: fallback)) ?? Data()
            return String(data: fallbackData, encoding: .utf8) ?? "{}"
        }
        return jsonString
    }

    nonisolated static func extractTextFromConversationResponse(_ json: String) -> String {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return json.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        if let content = obj["content"] as? [[String: Any]] {
            let texts = content.compactMap { $0["text"] as? String }
            if !texts.isEmpty { return texts.joined(separator: " ") }
        }

        if let candidates = obj["candidates"] as? [[String: Any]],
           let first = candidates.first,
           let content = first["content"] as? [String: Any],
           let parts = content["parts"] as? [[String: Any]] {
            let texts = parts.compactMap { $0["text"] as? String }
            if !texts.isEmpty { return texts.joined(separator: " ") }
        }

        if let text = obj["text"] as? String { return text }

        return json.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Stream Callback State

private final class StreamCallbackState: @unchecked Sendable {
    let continuation: AsyncThrowingStream<String, Error>.Continuation
    let doneSemaphore: DispatchSemaphore

    init(continuation: AsyncThrowingStream<String, Error>.Continuation,
         doneSemaphore: DispatchSemaphore) {
        self.continuation = continuation
        self.doneSemaphore = doneSemaphore
    }
}

/// Typed-event stream state used by `conversationSendTurnStreaming`.
///
/// Accessed only from the single LiteRT-LM callback thread, so the mutable
/// `buffer` / `yieldedToolCalls` fields don't need synchronization. The class
/// is `@unchecked Sendable` to satisfy the closure capture, matching the
/// existing `StreamCallbackState` pattern above.
private final class ConversationStreamState: @unchecked Sendable {
    let continuation: AsyncThrowingStream<LiteRTLMStreamEvent, Error>.Continuation
    let doneSemaphore: DispatchSemaphore
    var buffer: String = ""
    var yieldedToolCalls: Bool = false

    init(continuation: AsyncThrowingStream<LiteRTLMStreamEvent, Error>.Continuation,
         doneSemaphore: DispatchSemaphore) {
        self.continuation = continuation
        self.doneSemaphore = doneSemaphore
    }
}

// MARK: - Errors

public enum LiteRTLMError: LocalizedError {
    case modelNotFound
    case modelNotLoaded
    case engineCreationFailed(String)
    case inferenceFailure(String)
    case invalidToolSchema(toolName: String, detail: String)
    case noConversationOpen
    case malformedToolCallFromModel(rawJSON: String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound:
            "LiteRT-LM model file not found"
        case .modelNotLoaded:
            "LiteRT-LM model is not loaded — call load() first"
        case .engineCreationFailed(let detail):
            "Failed to create LiteRT-LM engine: \(detail)"
        case .inferenceFailure(let detail):
            "LiteRT-LM inference failed: \(detail)"
        case .invalidToolSchema(let name, let detail):
            "Invalid tool schema for '\(name)': \(detail)"
        case .noConversationOpen:
            "No persistent conversation open — call openConversation(...) first"
        case .malformedToolCallFromModel(let rawJSON):
            "Model produced a tool_call the wrapper could not parse. Raw: \(rawJSON)"
        }
    }
}
