import Foundation
import libSardonyx
import ArgumentParser

let options = try Options.parse()
var onnx = try Onnx_ModelProto(serializedData: Data(contentsOf: URL(fileURLWithPath: options.modelPath)))

var i = 0
while i < onnx.graph.node.count {
    let node = onnx.graph.node[i]
    if node.opType == "Add", let inputIdx = onnx.graph.input.firstIndex(where: { $0.name == node.input[0] }) {
        onnx.graph.node.remove(at: i)
        onnx.graph.input[inputIdx].name = node.output[0]
    }
    i += 1
}

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

