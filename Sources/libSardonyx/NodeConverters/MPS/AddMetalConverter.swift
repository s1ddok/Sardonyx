class AddMetalConverter: NodeConverter {
    let node: Onnx_NodeProto
    
    var graphInputs: [String] { Array(self.node.input[0...1]) }
    var graphOutputs: [String] { [self.node.output[0]] }
    
    
    required init(node: Onnx_NodeProto) {
        self.node = node
    }
    
    func contributeProperties(using context: GenerationContext) {
        context.sourceBuilder.add(line: "let add_\(self.node.name): MPSCNNAdd")
    }
    
    func contributeInit(using context: GenerationContext) {
        context.sourceBuilder.add(line: "self.add_\(self.node.name) = MPSCNNAdd(device: device)")
        context.sourceBuilder.add(line: "self.add_\(self.node.name).destinationImageAllocator = MPSTemporaryImage.defaultAllocator()")
    }
    
    func contributeImplementation(using context: GenerationContext) {
        context.sourceBuilder.add(line: "if _\(self.node.input[0]).featureChannels == 1 && _\(self.node.input[0]).featureChannels != _\(self.node.input[1]).featureChannels { self.add_\(self.node.name).primaryStrideInFeatureChannels = 0 } else { self.add_\(self.node.name).primaryStrideInFeatureChannels = 1 }")
        context.sourceBuilder.add(line: "if _\(self.node.input[1]).featureChannels == 1 && _\(self.node.input[0]).featureChannels != _\(self.node.input[1]).featureChannels { self.add_\(self.node.name).secondaryStrideInFeatureChannels = 0 } else { self.add_\(self.node.name).secondaryStrideInFeatureChannels = 1 }")
        
        context.sourceBuilder.add(line: "let _\(self.node.output[0]) = self.add_\(self.node.name).encode(commandBuffer: commandBuffer, primaryImage: _\(self.node.input[0]), secondaryImage: _\(self.node.input[1]))")
        
        let readCount = context.readCounts[self.node.output[0], default: 0]
        
        if readCount > 1 {
            context.sourceBuilder.add(line: "(_\(self.node.output[0]) as? MPSTemporaryImage)?.readCount = \(readCount)")
        }
        
    }
}
