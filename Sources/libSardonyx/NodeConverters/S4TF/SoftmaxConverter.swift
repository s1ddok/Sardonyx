class SoftmaxConverter: NodeConverter, ActivationConverter {
    let node: Onnx_NodeProto
    
    required init(node: Onnx_NodeProto) {
        self.node = node
    }
    
    func contributeImplementation(using context: GenerationContext) {
        let outputname = self.node.output[0]
        context.sourceBuilder.add(line: "let _\(outputname) = softmax(_\(self.node.input[0]))")
    }
    
    var closure: String { "softmax" }
    
    var output: String { self.node.output[0] }
}
