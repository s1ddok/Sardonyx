class UpsampleMetalConverter: NodeConverter {
    let node: Onnx_NodeProto
    
    enum Mode: String {
        case nearest = "nearest"
        case linear = "linear"
        
        var mpscnnType: String {
            switch self {
            case .nearest: return "MPSCNNUpsamplingNearest"
            case .linear: return "MPSCNNUpsamplingBilinear"
            }
        }
    }
    
    var mode: Mode
    
    required init(node: Onnx_NodeProto) {
        self.node = node
        
        self.mode = Mode(rawValue: String(data: node.attribute[0].s, encoding: .utf8) ?? "nearest") ?? .nearest
    }
    
    func contributeProperties(using context: GenerationContext) {
        context.sourceBuilder.add(line: "let upsample_\(self.node.name): \(self.mode.mpscnnType)")
    }
    func contributeInit(using context: GenerationContext) {
        
        let scales = context.tensors[self.node.input[1]]!.floats
        
        context.sourceBuilder.add(line: "self.upsample_\(self.node.name) = \(self.mode.mpscnnType)(device: device, integerScaleFactorX: \(Int(scales[3])), integerScaleFactorY: \(Int(scales[2])))")
    }
    
    func contributeImplementation(using context: GenerationContext) {
        context.sourceBuilder.add(line: "let _\(self.self.node.output[0]) = self.upsample_\(self.node.name).encode(commandBuffer: commandBuffer, sourceImage: _\(self.node.input[0]))")
        let readCount = context.readCounts[self.node.output[0], default: 0]
        
        if readCount > 1 {
            context.sourceBuilder.add(line: "(_\(self.node.output[0]) as? MPSTemporaryImage)?.readCount = \(readCount)")
        }
    }
}
