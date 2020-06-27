class ReshapeConverter: NodeConverter {
    let node: Onnx_NodeProto
    
    required init(node: Onnx_NodeProto) {
        self.node = node
    }
    
    func contributeImplementation(using context: GenerationContext) {
        let outputname = self.node.output[0]
        
        let shape = context.tensors[node.input[1]]!.int64Data.map(Int.init)
        context.sourceBuilder.add(line: "let _\(outputname) = _\(self.node.input[0]).reshaped(to: [\(shape.map(String.init).joined(separator: ", "))])")
    }
}
