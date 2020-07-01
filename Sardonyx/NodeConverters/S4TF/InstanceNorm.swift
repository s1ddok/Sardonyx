import Foundation

class InstanceNormConverter: NodeConverter {
    let node: Onnx_NodeProto
    
    var scaleOffset: Int = 0
    var BOffset: Int = 0
    
    var scaleLength: Int = 0
    var BLength: Int = 0
    
    var eps: Float
    
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
        context.sourceBuilder.add(line: "var layer_\(self.node.name): InstanceNorm<Float>")
    }
    
    func contributeInit(using context: GenerationContext) {
        context.sourceBuilder.add(line: "let offset_\(self.node.name) = Tensor<Float>(shape: [\(self.BLength)], scalars: UnsafeBufferPointer<Float>(start: data.advanced(by: \(self.BOffset)).assumingMemoryBound(to: Float.self), count: \(self.BLength)), on: device)")
        context.sourceBuilder.add(line: "let scale_\(self.node.name) = Tensor<Float>(shape: [\(self.scaleLength)], scalars: UnsafeBufferPointer<Float>(start: data.advanced(by: \(self.scaleOffset)).assumingMemoryBound(to: Float.self), count: \(self.scaleLength)), on: device)")
        
        context.sourceBuilder.add(line: "self.layer_\(self.node.name) = InstanceNorm<Float>(offset: offset_\(self.node.name), scale: scale_\(self.node.name), axis: -1, epsilon: \(self.eps))")
    }
    
    func contributeImplementation(using context: GenerationContext)  {
        let outputname = self.node.output[0]
        context.sourceBuilder.add(line: "let _\(outputname) = self.layer_\(self.node.name)(_\(self.node.input[0]))")
    }
}
