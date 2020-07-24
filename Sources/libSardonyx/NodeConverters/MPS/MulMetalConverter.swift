class MulMetalConverter: NodeConverter {
    let node: Onnx_NodeProto
    
    var graphInputs: [String] { Array(self.node.input[0...1]) }
    var graphOutputs: [String] { [self.node.output[0]] }
    
    
    required init(node: Onnx_NodeProto) {
        self.node = node
    }
    
    func contributeProperties(using context: GenerationContext) {
        context.sourceBuilder.add(line: "let mul_\(self.node.name): MPSCNNMultiply")
    }
    
    func contributeInit(using context: GenerationContext) {
        context.sourceBuilder.add(line: "self.mul_\(self.node.name) = MPSCNNMultiply(device: device)")
        context.sourceBuilder.add(line: "self.mul_\(self.node.name).destinationImageAllocator = MPSTemporaryImage.defaultAllocator()")
    }
    
    func contributeImplementation(using context: GenerationContext) {
        context.sourceBuilder.add(line: "if _\(self.node.input[0]).featureChannels == 1 && _\(self.node.input[0]).featureChannels != _\(self.node.input[1]).featureChannels { self.mul_\(self.node.name).primaryStrideInFeatureChannels = 0 } else { self.mul_\(self.node.name).primaryStrideInFeatureChannels = 1 }")
        context.sourceBuilder.add(line: "if _\(self.node.input[1]).featureChannels == 1 && _\(self.node.input[0]).featureChannels != _\(self.node.input[1]).featureChannels { self.mul_\(self.node.name).secondaryStrideInFeatureChannels = 0 } else { self.mul_\(self.node.name).secondaryStrideInFeatureChannels = 1 }")
        context.sourceBuilder.add(line: "let _\(self.node.output[0]) = self.mul_\(self.node.name).encode(commandBuffer: commandBuffer, primaryImage: _\(self.node.input[0]), secondaryImage: _\(self.node.input[1]))")
        
        let readCount = context.readCounts[self.node.output[0], default: 0]
        
        if readCount > 1 {
            context.sourceBuilder.add(line: "(_\(self.node.output[0]) as? MPSTemporaryImage)?.readCount = \(readCount)")
        }
        
    
    }
}
