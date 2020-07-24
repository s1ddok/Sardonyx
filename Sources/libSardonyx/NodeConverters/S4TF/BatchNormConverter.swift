import Foundation

class BatchNormConverter: NodeConverter {
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
        

        let scaleData = scale.floatData.withUnsafeBufferPointer { pointer -> Data in
            return Data(buffer: pointer)
        }
        let BData = B.floatData.withUnsafeBufferPointer { pointer -> Data in
            return Data(buffer: pointer)
        }
        let meanData = mean.floatData.withUnsafeBufferPointer { pointer -> Data in
            return Data(buffer: pointer)
        }
        let varData = variance.floatData.withUnsafeBufferPointer { pointer -> Data in
            return Data(buffer: pointer)
        }
        
        self.scaleOffset = context.add(data: scaleData)
        self.BOffset = context.add(data: BData)
        self.meanOffset = context.add(data: meanData)
        self.variangeOffset = context.add(data: varData)
        
        self.scaleLength = scale.floatData.count
        self.BLength = B.floatData.count
        self.meanLength = mean.floatData.count
        self.variangeLength = variance.floatData.count
    }
    
    func contributeProperties(using context: GenerationContext) {
        context.sourceBuilder.add(line: "var layer_\(self.node.name): BatchNorm<Float>")
    }
    
    func contributeInit(using context: GenerationContext) {
        context.sourceBuilder.add(line: "let offset_\(self.node.name) = Tensor<Float>(shape: [\(self.BLength)], scalars: UnsafeBufferPointer<Float>(start: data.advanced(by: \(self.BOffset)).assumingMemoryBound(to: Float.self), count: \(self.BLength)), on: device)")
        context.sourceBuilder.add(line: "let scale_\(self.node.name) = Tensor<Float>(shape: [\(self.scaleLength)], scalars: UnsafeBufferPointer<Float>(start: data.advanced(by: \(self.scaleOffset)).assumingMemoryBound(to: Float.self), count: \(self.scaleLength)), on: device)")
        context.sourceBuilder.add(line: "let mean_\(self.node.name) = Tensor<Float>(shape: [\(self.meanLength)], scalars: UnsafeBufferPointer<Float>(start: data.advanced(by: \(self.meanOffset)).assumingMemoryBound(to: Float.self), count: \(self.meanLength)), on: device)")
        context.sourceBuilder.add(line: "let var_\(self.node.name) = Tensor<Float>(shape: [\(self.variangeLength)], scalars: UnsafeBufferPointer<Float>(start: data.advanced(by: \(self.variangeOffset)).assumingMemoryBound(to: Float.self), count: \(self.variangeLength)), on: device)")
        
        context.sourceBuilder.add(line: "self.layer_\(self.node.name) = BatchNorm<Float>(axis: -1, momentum: \(self.momentum), offset: offset_\(self.node.name), scale: scale_\(self.node.name), epsilon: \(self.eps), runningMean: mean_\(self.node.name), runningVariance: var_\(self.node.name))")
    }
    
    func contributeImplementation(using context: GenerationContext)  {
        let outputname = self.node.output[0]
        context.sourceBuilder.add(line: "let _\(outputname) = self.layer_\(self.node.name)(_\(self.node.input[0]))")
    }
}
