class UpsampleConverter: NodeConverter {
    let node: Onnx_NodeProto
    
    enum Mode: String {
        case nearest = "nearest"
        case linear = "linear"
        
        var s4tf: String {
            switch self {
            case .nearest: return ".nearest"
            case .linear: return ".bilinear"
            }
        }
    }
    
    var mode: Mode
    
    required init(node: Onnx_NodeProto) {
        self.node = node
        
        self.mode = Mode(rawValue: String(data: node.attribute[0].s, encoding: .utf8) ?? "nearest") ?? .nearest
    }
    
    func contributeImplementation(using context: GenerationContext) {
        let outputname = self.node.output[0]
        
        let scales = context.tensors[self.node.input[1]]!.floats
        context.sourceBuilder.add(line: "let _\(outputname) = resize(images: _\(self.node.input[0]), size: (_\(self.node.input[0]).shape[1] * \(Int(scales[2])), _\(self.node.input[0]).shape[2] * \(Int(scales[3]))), method: \(self.mode.s4tf), antialias: false)")
    }
}
