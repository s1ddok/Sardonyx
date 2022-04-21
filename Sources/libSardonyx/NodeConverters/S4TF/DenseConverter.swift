import Foundation

class DenseConverter: NodeConverter, ActivationConverterInjectable {
    let node: Onnx_NodeProto
    let hasBias: Bool
    var weightOffset: Int = 0
    var inputChannels: Int = 0
    var outputChannels: Int = 0
    var biasOffset: Int? = nil
    
    var activationConverter: ActivationConverter? = nil
    
    required init(node: Onnx_NodeProto) {
        self.node = node
        self.hasBias = self.node.input.count > 2
    }
    
    func prepareData(using context: GenerationContext) throws {
        guard
            let weight = context.tensors[self.node.input[1]]
        else { fatalError() }
        
        var bias: Onnx_TensorProto?
        if self.hasBias {
            bias = context.tensors[node.input[2]]
        }
        
        let shouldTransposeB = self.node.attribute[3].i
        var weightScalars = weight.floatData
        if shouldTransposeB == 1 {
            weightScalars = weightScalars.transposed(to: [1, 0],
                                                     assuming: weight.dims.map(Int.init))
        }
        let weightData = weightScalars.withUnsafeBufferPointer { pointer -> Data in
            return Data(buffer: pointer)
        }
        let biasData = bias?.floatData.withUnsafeBufferPointer { pointer -> Data in
            return Data(buffer: pointer)
        }
        
        self.weightOffset = context.add(data: weightData)
        if shouldTransposeB == 1 {
            self.outputChannels = Int(weight.dims[0])
            self.inputChannels = Int(weight.dims[1])
        } else {
            self.outputChannels = Int(weight.dims[1])
            self.inputChannels = Int(weight.dims[0])
        }
        if let bd = biasData {
            self.biasOffset = context.add(data: bd)
        }
    }
    
    func contributeProperties(using context: GenerationContext) {
        context.sourceBuilder.add(line: "var layer_\(self.node.name): Dense<Float>")
    }
    
    func contributeInit(using context: GenerationContext) {
        context.sourceBuilder.add(line: "let weight_\(self.node.name) = Tensor<Float>(shape: [\(self.inputChannels), \(self.outputChannels)], scalars: UnsafeBufferPointer<Float>(start: data.advanced(by: \(self.weightOffset)).assumingMemoryBound(to: Float.self), count: \(self.inputChannels * self.outputChannels)), on: device)")
        
        if self.hasBias {
            context.sourceBuilder.add(line: "let bias_\(self.node.name) = Tensor<Float>(shape: [\(self.outputChannels)], scalars: UnsafeBufferPointer<Float>(start: data.advanced(by: \(self.biasOffset!)).assumingMemoryBound(to: Float.self), count: \(self.outputChannels)), on: device)")
        } else {
            context.sourceBuilder.add(line: "let bias_\(self.node.name): Tensor<Float>? = nil")
        }
        
        let activationClosure = self.activationConverter?.closure ?? "identity"
        context.sourceBuilder.add(line: "self.layer_\(self.node.name) = Dense<Float>(weight: weight_\(self.node.name), bias: bias_\(self.node.name), activation: \(activationClosure))")
    }
    
    func contributeImplementation(using context: GenerationContext) {
        let outputname = self.activationConverter?.output ?? self.node.output[0]
        context.sourceBuilder.add(line: "let _\(outputname) = self.layer_\(self.node.name)(_\(self.node.input[0]))")
    }
}
