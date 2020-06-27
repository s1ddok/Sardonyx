import Foundation

class GraphConverter {
    let context: GenerationContext
    let graph: Onnx_GraphProto
    let converters: [String: NodeConverter.Type]
    
    init(graph: Onnx_GraphProto) {
        self.graph = graph
        self.context = GenerationContext(graph: graph)
        self.converters = [
            "Conv": Conv2DConverter.self,
            "MaxPool": MaxPoolConverter.self,
            "Flatten": FlattenConverter.self,
            "Relu": RELUConverter.self,
            "Gemm": DenseConverter.self,
            "Dropout": DropoutConverter.self,
            "Add": AddConverter.self,
            "BatchNormalization": BatchNormConverter.self,
            "Reshape": ReshapeConverter.self,
            "GlobalAveragePool": GlobalAveragePoolConverter.self
        ]
    }
    
    func source() throws -> String {
        let converters = self.graph.node.map { self.converters[$0.opType]!.init(node: $0) }
        try converters.forEach { try $0.prepareData(using: self.context) }
        var source = "public struct \(self.graph.name): Layer {\n"
        for c in converters {
            source += c.contributeProperties(using: self.context)
        }
        
        source += "   init(data: UnsafeRawPointer) {\n"
        for c in converters {
            source += c.contributeInit(using: self.context)
        }
        source += "   }\n"
        
        source += "   @differentiable public func callAsFunction(_ _\(self.graph.input[0].name): Tensor<Float>) -> Tensor<Float> {\n"
        for c in converters {
            source += c.contributeImplementation(using: self.context)
        }
        source += "       return _\(self.graph.output[0].name)\n"
        source += "   }\n"
        source += "}\n"
        
        return source
    }
    
    func serialize(in directoryURL: URL, with name: String? = nil) throws {
        let source = try self.source()
        let swiftURL = directoryURL.appendingPathComponent("\(name ?? self.graph.name).swift")
        try source.write(to: swiftURL, atomically: true, encoding: .utf16)
        try self.context.globalDataBlob.write(to: directoryURL.appendingPathComponent("\(name ?? self.graph.name).data"))
    }
    
    
}
