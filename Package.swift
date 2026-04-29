// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LiteRTLMSwift",
    platforms: [
        .iOS(.v17)
    ],
    products: [
        .library(name: "LiteRTLMSwift", targets: ["LiteRTLMSwift"])
    ],
    targets: [
        .binaryTarget(
            name: "CLiteRTLM",
            path: "Frameworks/LiteRTLM.xcframework"
        ),
        .binaryTarget(
            name: "GemmaModelConstraintProvider",
            path: "Frameworks/GemmaModelConstraintProvider.xcframework"
        ),
        .binaryTarget(
            name: "LiteRtMetalAccelerator",
            path: "Frameworks/LiteRtMetalAccelerator.xcframework"
        ),
        .binaryTarget(
            name: "LiteRtTopKMetalSampler",
            path: "Frameworks/LiteRtTopKMetalSampler.xcframework"
        ),
        .target(
            name: "LiteRTLMSwift",
            dependencies: [
                "CLiteRTLM",
                "GemmaModelConstraintProvider",
                "LiteRtMetalAccelerator",
                "LiteRtTopKMetalSampler",
            ],
            path: "Sources/LiteRTLMSwift"
        ),
    ]
)
