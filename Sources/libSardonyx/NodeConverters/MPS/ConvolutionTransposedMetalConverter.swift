import Foundation

class ConvolutionTransposeMetalConverter: NodeConverter, FusableMetalNeuronInjectable, PadInjectable {
    func prepareData(using context: GenerationContext) throws {
        guard
            let weight = context.tensors[self.node.input[1]]
        else { fatalError() }
        
        var bias: Onnx_TensorProto?
        if self.node.input.count > 2 {
            bias = context.tensors[node.input[2]]
        }
        self.outputChannels = Int(weight.dims[1])
        self.inputChannels = Int(weight.dims[0])
        
        let weightData = weight.floats.reformatConvWeightToMPS(outputChannels: self.outputChannels, inputChannels: self.inputChannels, kernelHeight: self.kernel.height, kernelWidth: self.kernel.width, isTranspose: true)//transposed(to: [1, 2, 3, 0], assuming: weight.dims.map(Int.init))
            .withUnsafeBufferPointer { pointer -> Data in
            return Data(buffer: pointer)
        }
        let biasData = bias?.floats.withUnsafeBufferPointer { pointer -> Data in
            return Data(buffer: pointer)
        }
        
        self.weightOffset = context.add(data: weightData)

        if let bd = biasData {
            self.biasOffset = context.add(data: bd)
        }
    }
    
    func contributeProperties(using context: GenerationContext) {
        context.sourceBuilder.add(line: "var convolution_\(self.node.name): MPSCNNConvolutionTranspose")
    }
    
    func contributeInit(using context: GenerationContext) {
        if let neuron = self.neuron {
            context.sourceBuilder.add(line: neuron.descriptor)
        }
        
        let biasString = self.hasBias ? "data.advanced(by: \(self.biasOffset!)).assumingMemoryBound(to: Float.self)" : "nil"
        context.sourceBuilder.add(line: "let dataSource_\(self.node.name) = ConvolutionDataSource(weights: data.advanced(by: \(self.weightOffset)), bias: \(biasString), outputChannels: \(self.outputChannels), kernelHeight: \(self.kernel.height), kernelWidth: \(self.kernel.width), inputChannels: \(self.inputChannels), dilations: (\(self.dilations.height), \(self.dilations.width)), strides: (\(self.strides.height), \(self.strides.width)), groups: \(self.groups), neuron: \(self.neuron?.neuron ?? "nil"))")
        
        context.sourceBuilder.add(line: "self.convolution_\(self.node.name) = MPSCNNConvolutionTranspose(device: device, weights: dataSource_\(self.node.name))")
        context.sourceBuilder.add(line: "self.convolution_\(self.node.name).accumulatorPrecisionOption = .half")
        context.sourceBuilder.add(line: "self.convolution_\(self.node.name).destinationImageAllocator = MPSTemporaryImage.defaultAllocator()")
        
        if let pad = self.pad {
            let modeValue = pad.mode == "constant" ? ".zero" : ".mirror"
            context.sourceBuilder.add(line: "self.convolution_\(self.node.name).edgeMode = \(modeValue)")
            context.sourceBuilder.add(line: "self.convolution_\(self.node.name).padding = ONNXConvolutionPadding(kernel: (\(self.kernel.height), \(self.kernel.width)), strides: (\(self.strides.height), \(self.strides.width)), dilations: (\(self.dilations.height), \(self.dilations.width)), pads: (\(pad.yPadding.0), \(pad.yPadding.1), \(pad.xPadding.0), \(pad.xPadding.1)), outputPadding: (\(self.outputPadding.height), \(self.outputPadding.width)), isTranspose: true)")
        } else {
            context.sourceBuilder.add(line: "self.convolution_\(self.node.name).padding = ONNXConvolutionPadding(kernel: (\(self.kernel.height), \(self.kernel.width)), strides: (\(self.strides.height), \(self.strides.width)), dilations: (\(self.dilations.height), \(self.dilations.width)), pads: (\(pads.0), \(pads.1), \(pads.2), \(pads.3)), outputPadding: (\(self.outputPadding.height), \(self.outputPadding.width)), isTranspose: true)")
        }
        
    }
    
    func contributeImplementation(using context: GenerationContext) {
        let outputname = self.neuron?.output ?? self.node.output[0]
        context.sourceBuilder.add(line: "let _\(outputname) = self.convolution_\(self.node.name).encode(commandBuffer: commandBuffer, sourceImage: _\(self.pad?.node.input[0] ?? self.node.input[0]))")
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
    let pads: Pads
    
    
    var pad: PadMetalConverter? = nil
    var neuron: FusableMetalNeuron? = nil
    
    var graphInputs: [String] { [ self.pad?.node.input[0] ?? self.node.input[0]] }
    var graphOutputs: [String] { [ self.neuron?.output ?? self.node.output[0]] }
    
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
        self.pads = pads
    }
}
