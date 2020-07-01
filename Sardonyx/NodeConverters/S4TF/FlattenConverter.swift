class FlattenConverter: NodeConverter {
    let node: Onnx_NodeProto
    
    required init(node: Onnx_NodeProto) {
        self.node = node
    }
    
    func contributeProperties(using context: GenerationContext) {
        context.sourceBuilder.add(line: "var layer_\(self.node.name) = Flatten<Float>()")
    }
    
    func contributeImplementation(using context: GenerationContext) {
        let outputname = self.node.output[0]
        context.sourceBuilder.add(line: "let flattened_\(self.node.input[0]): Tensor<Float>")
        context.sourceBuilder.scope(with: "if _\(self.node.input[0]).rank == 4") {
            context.sourceBuilder.add(line: "flattened_\(self.node.input[0]) = _\(self.node.input[0]).transposed(permutation: [0, 3, 2, 1])")
        }
        context.sourceBuilder.add(line: "else { flattened_\(self.node.input[0]) = _\(self.node.input[0]) }")

        context.sourceBuilder.add(line: "let _\(outputname) = self.layer_\(self.node.name)(flattened_\(self.node.input[0]))")
    }
}
