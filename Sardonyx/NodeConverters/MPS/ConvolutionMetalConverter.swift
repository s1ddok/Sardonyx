import Foundation

class ConvolutionMetalConverter: NodeConverter, FusableMetalNeuronInjectable, PadInjectable, BatchNormInjectable {
    func prepareData(using context: GenerationContext) throws {
        guard
            let weight = context.tensors[self.node.input[1]]
        else { fatalError() }
        
        var bias: Onnx_TensorProto?
        if self.node.input.count > 2 {
            bias = context.tensors[node.input[2]]
        }
        
        let weightFloats = weight.floats.transposed(to: [0, 2, 3, 1], assuming: weight.dims.map(Int.init))
            
        let weightData = weightFloats.data(with: context.shouldConvertWeightsToFloat16 ? .half : .full)
        let biasData = bias?.floats.data(with: .full)
        
        self.weightOffset = context.add(data: weightData)
        self.outputChannels = Int(weight.dims[0])
        self.inputChannels = Int(weight.dims[1])
        if let bd = biasData {
            self.biasOffset = context.add(data: bd)
        }
        
        try self.batchNorm?.prepareData(using: context)
    }
    
    func contributeProperties(using context: GenerationContext) {
        context.sourceBuilder.add(line: "var convolution_\(self.node.name): MPSCNNConvolution")
    }
    
    func contributeInit(using context: GenerationContext) {
        if let neuron = self.neuron {
            context.sourceBuilder.add(line: neuron.descriptor)
        }
        
        let biasString = self.hasBias ? "data.advanced(by: \(self.biasOffset!)).assumingMemoryBound(to: Float.self)" : "nil"
        var batchNormParameters = ""
        if let bn = self.batchNorm {
            batchNormParameters = ", mean: data.advanced(by: \(bn.meanOffset)).assumingMemoryBound(to: Float.self), variance: data.advanced(by: \(bn.variangeOffset)).assumingMemoryBound(to: Float.self), gamma: data.advanced(by: \(bn.scaleOffset)).assumingMemoryBound(to: Float.self), beta: data.advanced(by: \(bn.BOffset)).assumingMemoryBound(to: Float.self), epsilon: \(bn.eps)"
        }
        context.sourceBuilder.add(line: "let dataSource_\(self.node.name) = ConvolutionDataSource(weights: data.advanced(by: \(self.weightOffset)), weightDataType: \(context.shouldConvertWeightsToFloat16 ? ".float16" : ".float32"), bias: \(biasString), outputChannels: \(self.outputChannels), kernelHeight: \(self.kernel.height), kernelWidth: \(self.kernel.width), inputChannels: \(self.inputChannels), dilations: (\(self.dilations.height), \(self.dilations.width)), strides: (\(self.strides.height), \(self.strides.width)), groups: \(self.groups), neuron: \(self.neuron?.neuron ?? "nil")\(batchNormParameters))")
        
        context.sourceBuilder.add(line: "self.convolution_\(self.node.name) = MPSCNNConvolution(device: device, weights: dataSource_\(self.node.name))")
        if context.shouldConvertWeightsToFloat16 {
            context.sourceBuilder.add(line: "self.convolution_\(self.node.name).accumulatorPrecisionOption = .half")
        }
        context.sourceBuilder.add(line: "self.convolution_\(self.node.name).destinationImageAllocator = MPSTemporaryImage.defaultAllocator()")
        
        if let pad = self.pad {
            let modeValue = pad.mode == "constant" ? ".zero" : ".mirror"
            context.sourceBuilder.add(line: "self.convolution_\(self.node.name).edgeMode = \(modeValue)")
            context.sourceBuilder.add(line: "self.convolution_\(self.node.name).padding = ONNXConvolutionPadding(kernel: (\(self.kernel.height), \(self.kernel.width)), strides: (\(self.strides.height), \(self.strides.width)), dilations: (\(self.dilations.height), \(self.dilations.width)), pads: (\(pad.yPadding.0), \(pad.yPadding.1), \(pad.xPadding.0), \(pad.xPadding.1)), outputPadding: (\(self.outputPadding.height), \(self.outputPadding.width)), isTranspose: false)")
        } else {
            context.sourceBuilder.add(line: "self.convolution_\(self.node.name).padding = MPSNNDefaultPadding(method: .\(self.padding))")
        }
        
    }
    
    func contributeImplementation(using context: GenerationContext) {
        context.sourceBuilder.add(line: "let _\(self.graphOutputs[0]) = self.convolution_\(self.node.name).encode(commandBuffer: commandBuffer, sourceImage: _\(self.pad?.node.input[0] ?? self.node.input[0]))")
    }
    
    let node: Onnx_NodeProto
    let hasBias: Bool
    var weightOffset: Int = 0
    var inputChannels: Int = 0
    var outputChannels: Int = 0
    var biasOffset: Int? = nil
    
    let kernel: Kernel
    let dilations: Dilations
    let strides: Strides
    let groups: Int
    let padding: String
    let outputPadding: Padding
    
    var pad: PadMetalConverter? = nil
    var batchNorm: BatchNormMetalConverter? = nil
    var neuron: FusableMetalNeuron? = nil
    
    var graphInputs: [String] { [ self.pad?.node.input[0] ?? self.node.input[0]] }
    var graphOutputs: [String] { [ self.batchNorm?.node.output[0] ?? self.neuron?.output ?? self.node.output[0]] }
    
    required init(node: Onnx_NodeProto) {
        self.node = node
        self.hasBias = node.input.count > 2
        var kernel: Kernel = (1, 1)
        var dilations: Dilations = (1, 1)
        var strides: Strides = (1, 1)
        var groups: Int = 1
        var pads: Pads = (0, 0, 0, 0)
        var outputPadding = Padding(height: 0, width: 0)
        
        for attr in node.attribute {
            switch attr.name {
            case "dilations":
                dilations = (Int(attr.ints[0]), Int(attr.ints[1]))
            case "strides":
                strides = (Int(attr.ints[0]), Int(attr.ints[1]))
            case "group":
                groups = Int(attr.i)
            case "pads":
                pads = (Int(attr.ints[0]), Int(attr.ints[1]), Int(attr.ints[2]), Int(attr.ints[3]))
            case "kernel_shape":
                kernel = (Int(attr.ints[0]), Int(attr.ints[1]))
            case "output_padding":
                outputPadding = (Int(attr.ints[0]), Int(attr.ints[1]))
            default:
                break
            }
        }
        
        if pads == (0, 0, 0, 0) {
            self.padding = "validOnly"
        } else if pads == (1, 1, 1, 1) {
            self.padding = "sizeSame"
        } else {
            fatalError("Please implement logic for padding: \(pads)")
        }
        
        self.kernel = kernel
        self.dilations = dilations
        self.strides = strides
        self.groups = groups
        self.outputPadding = outputPadding
    }
}
