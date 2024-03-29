import Foundation

public class MetalGraphConverter {
    let context: GenerationContext
    let graph: Onnx_GraphProto
    let converters: [String: NodeConverter.Type]
    
    public enum InputType {
        case mpsImage, mtlTexture, mtlBuffer
        
        var stringRepresentation: String {
            switch self {
            case .mpsImage: return "MPSImage"
            case .mtlBuffer: return "MTLBuffer"
            case .mtlTexture: return "MTLTexture"
            }
        }
    }
    
    private var userInputTypes: [String: InputType]
    
    public init(graph: Onnx_GraphProto,
                userConverters: [String: NodeConverter.Type] = [:],
                userInputTypes: [String: InputType] = [:],
                shouldConvertWeightsToFloat16: Bool = false,
                shouldConvertConstantsToFloat16: Bool = false) {
        self.userInputTypes = userInputTypes
        let context = GenerationContext(graph: graph)
        context.shouldConvertWeightsToFloat16 = shouldConvertWeightsToFloat16
        context.shouldConvertConstantsToFloat16 = shouldConvertConstantsToFloat16
        var constantFreeGraph = graph
        constantFreeGraph.node = constantFreeGraph.node.filter { node in
            if node.opType == "Constant" {
                for output in node.output {
                    context.register(tensor: node.attribute[0].t, with: output)
                }
                
                return false
            }
            
            return true
        }

        self.context = context
        self.graph = constantFreeGraph
        self.converters = [
            "Conv": ConvolutionMetalConverter.self,
            "Relu": RELUMetalConverter.self,
            "Add": AddMetalConverter.self,
            "Sigmoid": SigmoidMetalConverter.self,
            "Pad": PadMetalConverter.self,
            "InstanceNormalization": InstanceNormalizationMetalConverter.self,
            "Constant": ConstantConverter.self,
            "ConvTranspose": ConvolutionTransposeMetalConverter.self,
            "Tanh": TanhMetalConverter.self,
            "Mul": MulMetalConverter.self,
            "BatchNormalization": BatchNormMetalConverter.self,
            "Upsample": UpsampleMetalConverter.self,
            "LeakyRelu": LeakyRELUMetalConverter.self
        ].merging(userConverters, uniquingKeysWith: { $1 })
    }
    
    func source(modelName: String? = nil) throws -> String {
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
            
            if
                let bnInjectable = converters[i] as? BatchNormInjectable,
                converters[i].graphOutputs.count == 1,
                let subsequentBN = converters[i + 1] as? BatchNormMetalConverter
            {
                bnInjectable.batchNorm = subsequentBN
                converters.remove(at: i + 1)
            }
            
            guard i < converters.count - 1 else { break }
            
            if
                let neuronInjectable = converters[i] as? FusableMetalNeuronInjectable,
                let subsequentNeuron = converters[i + 1] as? FusableMetalNeuron
            {
                neuronInjectable.neuron = subsequentNeuron
                converters.remove(at: i + 1)
            }
            
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
        
        sourceBuilder.scope(with: "public class \(modelName ?? self.graph.name.validIdentifier)") {
            let userInputs = self.graph
                .input
                .filter { self.context.tensors[$0.name] == nil }
            
            if self.graph.output.count > 1 {
                let outputsTuple = "\(self.graph.output.map { "_\($0.name.validIdentifier): MPSImage" }.joined(separator: ", ")))"
                sourceBuilder.add(line: "public typealias Output = (\(outputsTuple))")
            } else {
                sourceBuilder.add(line: "public typealias Output = MPSImage")
            }
            
            var constantInputs = Set<String>()
            for c in converters {
                c.contributeProperties(using: self.context)
                
                for input in c.graphInputs {
                    if let _ = self.context.tensors[input], !constantInputs.contains(input) {
                        sourceBuilder.add(line: "var _\(input): MPSImage")
                        constantInputs.insert(input)
                    }
                }
            }
            sourceBuilder.blankLine()
            sourceBuilder.scope(with: "init(data: UnsafeMutableRawPointer, device: MTLDevice)") {
                for ci in constantInputs {
                    let tensor = self.context.tensors[ci]!
                    let floats = tensor.floats
                    
                    switch tensor.dims.count {
                    case 4:
                        let constantData = floats.data(with: context.shouldConvertConstantsToFloat16 ? .half : .full)
                        let channelFormat = context.shouldConvertConstantsToFloat16 ? ".float16" : ".float32"
                        let constantOffset = context.add(data: constantData)
                            
                        sourceBuilder.add(line: "self._\(ci) = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(channelFormat: \(channelFormat), width: \(Int(tensor.dims[3])), height: \(Int(tensor.dims[2])), featureChannels: \(Int(tensor.dims[1]))))")
                        sourceBuilder.add(line: "self._\(ci).writeBytes(data.advanced(by: \(constantOffset)), dataLayout: .featureChannelsxHeightxWidth, imageIndex: 0)")
                    default: ()
                    }

                }
                
                for c in converters {
                    c.contributeInit(using: self.context)
                }
            }
            sourceBuilder.blankLine()
            
            sourceBuilder.scope(with: "public func encode(commandBuffer: MTLCommandBuffer, \(userInputs.map { "_\($0.name.validIdentifier): \((self.userInputTypes[$0.name] ?? .mpsImage).stringRepresentation)" }.joined(separator: ", "))) -> Output") {
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
    
    public func serialize(in directoryURL: URL, with name: String? = nil) throws {
        let source = try self.source(modelName: name)
        let swiftURL = directoryURL.appendingPathComponent("\(name ?? self.graph.name).swift", isDirectory: false)
        try source.write(to: swiftURL, atomically: true, encoding: .utf16)
        try self.context.globalDataBlob.write(to: directoryURL.appendingPathComponent("\(name ?? self.graph.name).data"))
    }
    
    
}

extension String {
    var validIdentifier: String {
        return self.replacingOccurrences(of: "-", with: "_").replacingOccurrences(of: ".", with: "_")
    }
}
