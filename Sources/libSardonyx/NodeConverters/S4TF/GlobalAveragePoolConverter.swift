class GlobalAveragePoolConverter: NodeConverter {
    let node: Onnx_NodeProto
    
    required init(node: Onnx_NodeProto) {
        self.node = node
    }
    
    func contributeImplementation(using context: GenerationContext) {
        let outputname = self.node.output[0]
        context.sourceBuilder.add(line: "let _\(outputname) = _\(self.node.input[0]).mean(alongAxes: [1, 2])")
    }
}
