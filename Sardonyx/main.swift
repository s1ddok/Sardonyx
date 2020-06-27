import Foundation

let onnx = try Onnx_ModelProto(serializedData: Data(contentsOf: URL(fileURLWithPath: "/Users/av/Downloads/mobilenetv2-7.onnx")))

let converter = GraphConverter(graph: onnx.graph)
try! converter.serialize(in: URL(fileURLWithPath: "/Users/av/Downloads/", isDirectory: true))
