class AddConverter: NodeConverter {
    let node: Onnx_NodeProto
    
    var graphInputs: [String] { Array(self.node.input[0...1]) }
    
    required init(node: Onnx_NodeProto) {
        self.node = node
    }
    
    func contributeImplementation(using context: GenerationContext) {
        let outputname = self.node.output[0]
        context.sourceBuilder.add(line: "let _\(outputname) = _\(self.node.input[0]) + _\(self.node.input[1])")
    }
}
