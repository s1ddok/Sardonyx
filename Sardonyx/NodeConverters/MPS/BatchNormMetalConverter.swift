import Foundation

class BatchNormMetalConverter: NodeConverter {
    let node: Onnx_NodeProto
    
    var scaleOffset: Int = 0
    var BOffset: Int = 0
    var meanOffset: Int = 0
    var variangeOffset: Int = 0
    
    var scaleLength: Int = 0
    var BLength: Int = 0
    var meanLength: Int = 0
    var variangeLength: Int = 0
    
    var momentum: Float
    var eps: Float
    var graphInputs: [String] { [self.node.input[0]] }
    var graphOutputs: [String] { [self.node.output[0]] }
    required init(node: Onnx_NodeProto) {
        self.node = node
        self.eps = node.attribute[0].f
        self.momentum = node.attribute[1].f
    }
    
    func prepareData(using context: GenerationContext) throws {
        guard
            let scale = context.tensors[self.node.input[1]],
            let B = context.tensors[self.node.input[2]],
            let mean = context.tensors[self.node.input[3]],
            let variance = context.tensors[self.node.input[4]]
        else { fatalError() }
        

        let scaleData = scale.floats.withUnsafeBufferPointer { pointer -> Data in
            return Data(buffer: pointer)
        }
        let BData = B.floats.withUnsafeBufferPointer { pointer -> Data in
            return Data(buffer: pointer)
        }
        let meanData = mean.floats.withUnsafeBufferPointer { pointer -> Data in
            return Data(buffer: pointer)
        }
        let varData = variance.floats.withUnsafeBufferPointer { pointer -> Data in
            return Data(buffer: pointer)
        }
        
        self.scaleOffset = context.add(data: scaleData)
        self.BOffset = context.add(data: BData)
        self.meanOffset = context.add(data: meanData)
        self.variangeOffset = context.add(data: varData)
        
        self.scaleLength = scale.floats.count
        self.BLength = B.floats.count
        self.meanLength = mean.floats.count
        self.variangeLength = variance.floats.count
    }
    
    func contributeProperties(using context: GenerationContext) {
        context.sourceBuilder.add(line: "var batchNorm_\(self.node.name): MPSCNNBatchNormalization")
    }
    
    func contributeInit(using context: GenerationContext) {
        context.sourceBuilder.add(line: "let dataSource_\(self.node.name) = BatchNormDataSource(channels: \(self.BLength), mean: data.advanced(by: \(self.meanOffset)).assumingMemoryBound(to: Float.self), variance: data.advanced(by: \(self.variangeOffset)).assumingMemoryBound(to: Float.self), gammas: data.advanced(by: \(self.scaleOffset)).assumingMemoryBound(to: Float.self), betas: data.advanced(by: \(self.BOffset)).assumingMemoryBound(to: Float.self))")
        context.sourceBuilder.add(line: "self.batchNorm_\(self.node.name) = MPSCNNBatchNormalization(device: device, dataSource: dataSource_\(self.node.name))")
        context.sourceBuilder.add(line: "self.batchNorm_\(self.node.name).destinationImageAllocator = MPSTemporaryImage.defaultAllocator()")
    }
    
    func contributeImplementation(using context: GenerationContext)  {
        context.sourceBuilder.add(line: "let _\(self.node.output[0]) = self.batchNorm_\(self.node.name).encode(commandBuffer: commandBuffer, sourceImage: _\(self.node.input[0]))")
    }
}
