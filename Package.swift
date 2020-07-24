// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Sardonyx",
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", .upToNextMinor(from: "0.2.0")),
        .package(name: "SwiftProtobuf",
                  url: "https://github.com/apple/swift-protobuf.git",
                  .exact("1.7.0"))
    ],
    targets: [
        .target(name: "Sardonyx", dependencies: ["libSardonyx", .product(name: "ArgumentParser", package: "swift-argument-parser")]),
        .target(
            name: "libSardonyx",
            dependencies: ["SwiftProtobuf"]),
    ]
)
