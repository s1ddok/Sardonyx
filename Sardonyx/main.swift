import Foundation

let options = try Options.parse()
let onnx = try Onnx_ModelProto(serializedData: Data(contentsOf: URL(fileURLWithPath: options.modelPath)))

switch options.targetPlatform {
case .s4tf:
    let converter = GraphConverter(graph: onnx.graph)
    try converter.serialize(in: URL(fileURLWithPath: options.outputDirectory, isDirectory: true),
                            with: options.modelName)
case .mps:
    let converter = MetalGraphConverter(graph: onnx.graph)
    try converter.serialize(in: URL(fileURLWithPath: options.outputDirectory, isDirectory: true),
                            with: options.modelName)
}

