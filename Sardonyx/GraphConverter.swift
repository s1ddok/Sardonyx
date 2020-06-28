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
            "ConvTranspose": ConvTranspose2DConverter.self,
            "LeakyRelu": LeakyRELUConverter.self,
            "Tanh": TanhConverter.self,
            "Upsample": UpsampleConverter.self,
            "Mul": MulConverter.self
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
        sourceBuilder.add(line: "// This file is auto-generated, edit it with causion")
        sourceBuilder.add(line: "import TensorFlow")
        sourceBuilder.blankLine()
        
        sourceBuilder.scope(with: "public struct \(self.graph.name): Layer") {
            let userInputs = self.graph
                .input
                .filter { self.context.tensors[$0.name] == nil }
            
            if userInputs.count > 1 {
                sourceBuilder.scope(with: "public struct Input: Differentiable") {
                    userInputs.forEach { sourceBuilder.add(line: "var _\($0.name): Tensor<Float>") }
                }
            } else {
                sourceBuilder.add(line: "public typealias Input = Tensor<Float>")
            }

            if self.graph.output.count > 1 {
                let outputsTuple = "\(self.graph.output.map { "_\($0.name): Tensor<Float>" }.joined(separator: ", ")))"
                sourceBuilder.add(line: "public typealias Output = (\(outputsTuple))")
            } else {
                sourceBuilder.add(line: "public typealias Output = Tensor<Float>")
            }
            
            for c in converters {
                c.contributeProperties(using: self.context)
                
                for input in c.graphInputs {
                    if let _ = self.context.tensors[input] {
                        sourceBuilder.add(line: "var _\(input): Tensor<Float>")
                    }
                }
            }
            
            sourceBuilder.blankLine()
            sourceBuilder.scope(with: "init(data: UnsafeRawPointer, device: Device)") {
                for c in converters {
                    c.contributeInit(using: self.context)
                    
                    for input in c.graphInputs {
                        if let constantInput = self.context.tensors[input] {
                            let floats = constantInput.floats
                            switch constantInput.dims.count {
                            case 0, 1:
                                let constantData = floats.withUnsafeBufferPointer { pointer -> Data in
                                    return Data(buffer: pointer)
                                }
                                
                                let constantOffset = self.context.add(data: constantData)
                                
                                self.context.sourceBuilder.add(line: "self._\(input) = Tensor<Float>(shape: [\(constantInput.dims[safe: 0] ?? 1)], scalars: UnsafeBufferPointer<Float>(start: data.advanced(by: \(constantOffset)).assumingMemoryBound(to: Float.self), count: \(constantInput.dims[safe: 0] ?? 1)), on: device)")
                            case 4:
                                let constantData = floats.transposed(to: [0, 2, 3, 1], assuming: constantInput.dims.map(Int.init)).withUnsafeBufferPointer { pointer -> Data in
                                    return Data(buffer: pointer)
                                }
                                
                                let constantOffset = self.context.add(data: constantData)
                                
                                self.context.sourceBuilder.add(line: "self._\(input) = Tensor<Float>(shape: [\(constantInput.dims[0]), \(constantInput.dims[2]), \(constantInput.dims[3]), \(constantInput.dims[1])], scalars: UnsafeBufferPointer<Float>(start: data.advanced(by: \(constantOffset)).assumingMemoryBound(to: Float.self), count: \(Int(constantInput.dims.reduce(1, *)))), on: device)")
                            default: fatalError("Constants with rank \(constantInput.dims.count) are not supported")
                            }
                        }
                    }
                }
            }
            sourceBuilder.blankLine()
            
            sourceBuilder.scope(with: "@differentiable public func callAsFunction(_ input: Input) -> Output") {
                for userInput in userInputs {
                    sourceBuilder.add(line: "let _\(userInput.name) = input._\(userInput.name)")
                }
                
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
