import Foundation

class MetalGraphConverter {
    let context: GenerationContext
    let graph: Onnx_GraphProto
    let converters: [String: NodeConverter.Type]
    
    init(graph: Onnx_GraphProto) {
        self.graph = graph
        self.context = GenerationContext(graph: graph)
        self.converters = [
            "Conv": ConvolutionMetalConverter.self,
            "Relu": RELUMetalConverter.self,
            "Add": AddMetalConverter.self,
            "Sigmoid": SigmoidMetalConverter.self,
            "Pad": PadMetalConverter.self,
            "InstanceNormalization": InstanceNormalizationMetalConverter.self,
            "Constant": ConstantConverter.self,
            "ConvTranspose": ConvolutionTransposeMetalConverter.self,
        ]
    }
    
    func source() throws -> String {
        var converters = self.graph.node.compactMap { self.converters[$0.opType]!.init(node: $0) }
        
        var i = 0
        while i < converters.count - 1 {
            defer {
                i += 1
            }
            
            if let pad = converters[i] as? PadMetalConverter,
               let subsequentOp = converters[i + 1] as? PadInjectable {
                converters.remove(at: i)
                subsequentOp.pad = pad
            }
            
            guard i < converters.count - 1 else { break }
            
            guard
                let neuronInjectable = converters[i] as? FusableMetalNeuronInjectable,
                let subsequentNeuron = converters[i + 1] as? FusableMetalNeuron
            else {
                continue
            }
            
            neuronInjectable.neuron = subsequentNeuron
            converters.remove(at: i + 1)
        }
        
        for c in converters {
            for input in c.graphInputs {
                self.context.readCounts[input, default: 0] += 1
            }
        }
        
        try converters.forEach { try $0.prepareData(using: self.context) }
        
        let sourceBuilder = self.context.sourceBuilder
        sourceBuilder.add(line: "// This file is auto-generated, edit it with causion")
        sourceBuilder.add(line: "import Metal")
        sourceBuilder.add(line: "import MetalPerformanceShaders")
        sourceBuilder.blankLine()
        
        sourceBuilder.scope(with: "public class \(self.graph.name.validIdentifier)") {
            let userInputs = self.graph
                .input
                .filter { self.context.tensors[$0.name] == nil }
            
            if self.graph.output.count > 1 {
                let outputsTuple = "\(self.graph.output.map { "_\($0.name.validIdentifier): MPSImage" }.joined(separator: ", ")))"
                sourceBuilder.add(line: "public typealias Output = (\(outputsTuple))")
            } else {
                sourceBuilder.add(line: "public typealias Output = MPSImage")
            }
            
            for c in converters {
                c.contributeProperties(using: self.context)
                
                for input in c.graphInputs {
                    if let _ = self.context.tensors[input] {
                        sourceBuilder.add(line: "var _\(input): MPSImage")
                    }
                }
            }
            
            sourceBuilder.blankLine()
            sourceBuilder.scope(with: "init(data: UnsafeMutableRawPointer, device: MTLDevice)") {
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
            
            sourceBuilder.scope(with: "public func encode(commandBuffer: MTLCommandBuffer, \(userInputs.map { "_\($0.name.validIdentifier): MPSImage" }.joined(separator: ", "))) -> Output") {
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

extension String {
    var validIdentifier: String {
        return self.replacingOccurrences(of: "-", with: "_").replacingOccurrences(of: ".", with: "_")
    }
}
