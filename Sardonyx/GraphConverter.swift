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
            "GlobalAveragePool": GlobalAveragePoolConverter.self,
            "Sigmoid": SigmoidConverter.self,
            "Softmax": SoftmaxConverter.self,
            "Pad": PadConverter.self,
            "InstanceNormalization": InstanceNormConverter.self,
            "Constant": ConstantConverter.self,
            "ConvTranspose": ConvTranspose2DConverter.self
        ]
    }
    
    func source() throws -> String {
        var converters = self.graph.node.compactMap { self.converters[$0.opType]!.init(node: $0) }
        
        var i = 0
        while i < converters.count - 1 {
            defer {
                i += 1
            }
            guard
                let activationInjectable = converters[i] as? ActivationConverterInjectable,
                let subsequentActivation = converters[i + 1] as? ActivationConverter
            else {
                continue
            }
            
            activationInjectable.activationConverter = subsequentActivation
            converters.remove(at: i + 1)
        }
        
        try converters.forEach { try $0.prepareData(using: self.context) }
        
        let sourceBuilder = self.context.sourceBuilder
        sourceBuilder.add(line: "// This is file is auto-generated, edit it with causion")
        sourceBuilder.add(line: "import TensorFlow")
        sourceBuilder.blankLine()
        
        sourceBuilder.scope(with: "public struct \(self.graph.name): Layer") {
            for c in converters {
                c.contributeProperties(using: self.context)
            }
            
            sourceBuilder.blankLine()
            sourceBuilder.scope(with: "init(data: UnsafeRawPointer, device: Device)") {
                for c in converters {
                    c.contributeInit(using: self.context)
                }
            }
            sourceBuilder.blankLine()
            
            let inputs = self.graph
                .input
                .filter { self.context.tensors[$0.name] == nil }
                .map { "_ _\($0.name): Tensor<Float>" }.joined(separator: ", ")
            let outputs: String
            if self.graph.output.count > 1 {
                outputs = "(\(self.graph.output.map { "_\($0.name): Tensor<Float>" }.joined(separator: ", ")))"
            } else {
                outputs = "Tensor<Float>"
            }
            
            sourceBuilder.scope(with: "@differentiable public func callAsFunction(\(inputs)) -> \(outputs)") {
                for c in converters {
                    c.contributeImplementation(using: self.context)
                }
                
                let outputsToReturn = self.graph.output.map { "_\($0.name)" }.joined(separator: ", ")
                if self.graph.output.count > 1 {
                    sourceBuilder.add(line: "return (\(outputsToReturn))")
                } else {
                    sourceBuilder.add(line: "return \(outputsToReturn)")
                }
            }
        }
        
        return sourceBuilder.result
    }
    
    func serialize(in directoryURL: URL, with name: String? = nil) throws {
        let source = try self.source()
        let swiftURL = directoryURL.appendingPathComponent("\(name ?? self.graph.name).swift")
        try source.write(to: swiftURL, atomically: true, encoding: .utf16)
        try self.context.globalDataBlob.write(to: directoryURL.appendingPathComponent("\(name ?? self.graph.name).data"))
    }
    
    
}
