import Foundation

class InstanceNormalizationMetalConverter: NodeConverter {
    let node: Onnx_NodeProto
    
    var scaleOffset: Int = 0
    var BOffset: Int = 0
    
    var scaleLength: Int = 0
    var BLength: Int = 0
    
    var eps: Float
    
    var graphInputs: [String] { [self.node.input[0]] }
    var graphOutputs: [String] { [self.node.output[0]] }
    
    required init(node: Onnx_NodeProto) {
        self.node = node
        self.eps = node.attribute[0].f
    }
    
    func prepareData(using context: GenerationContext) throws {
        guard
            let scale = context.tensors[self.node.input[1]],
            let B = context.tensors[self.node.input[2]]
        else { fatalError() }
        

        let scaleData = scale.floats.withUnsafeBufferPointer { pointer -> Data in
            return Data(buffer: pointer)
        }
        let BData = B.floats.withUnsafeBufferPointer { pointer -> Data in
            return Data(buffer: pointer)
        }
        
        self.scaleOffset = context.add(data: scaleData)
        self.BOffset = context.add(data: BData)
        
        self.scaleLength = scale.floats.count
        self.BLength = B.floats.count
    }
    
    func contributeProperties(using context: GenerationContext) {
        context.sourceBuilder.add(line: "var instanceNorm_\(self.node.name): MPSCNNInstanceNormalization")
    }
    
    func contributeInit(using context: GenerationContext) {
        context.sourceBuilder.add(line: "let dataSource_\(self.node.name) = InstanceNormDataSource(channels: \(self.BLength), gammas: data.advanced(by: \(self.scaleOffset)).assumingMemoryBound(to: Float.self), betas: data.advanced(by: \(self.BOffset)).assumingMemoryBound(to: Float.self))")
        context.sourceBuilder.add(line: "self.instanceNorm_\(self.node.name) = MPSCNNInstanceNormalization(device: device, dataSource: dataSource_\(self.node.name))")
        context.sourceBuilder.add(line: "self.instanceNorm_\(self.node.name).destinationImageAllocator = MPSTemporaryImage.defaultAllocator()")
    }
    
    func contributeImplementation(using context: GenerationContext)  {
        context.sourceBuilder.add(line: "let _\(self.node.output[0]) = self.instanceNorm_\(self.node.name).encode(commandBuffer: commandBuffer, sourceImage: _\(self.node.input[0]))")
    }
}
