class LeakyRELUConverter: NodeConverter, ActivationConverter {
    let node: Onnx_NodeProto
    let alpha: Float
    
    required init(node: Onnx_NodeProto) {
        self.node = node
        self.alpha = node.attribute[0].f
    }
    
    func contributeImplementation(using context: GenerationContext) {
        let outputname = self.node.output[0]
        context.sourceBuilder.add(line: "let _\(outputname) = leakyRelu(_\(self.node.input[0]), alpha: \(self.alpha))")
    }
    
    var closure: String { "{ leakyRelu($0, alpha: \(self.alpha)) }" }
    
    var output: String { self.node.output[0] }
}
