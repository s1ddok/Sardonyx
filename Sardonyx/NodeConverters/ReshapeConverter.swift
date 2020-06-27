class ReshapeConverter: NodeConverter {
    let node: Onnx_NodeProto
    var shape: [Int] = []
    required init(node: Onnx_NodeProto) {
        self.node = node
    }
    
    func prepareData(using context: GenerationContext) throws {
        self.shape = context.tensors[node.input[1]]!.int64Data.map(Int.init)
    }
    
    func contributeProperties(using: GenerationContext) -> String {
        ""
    }
    
    func contributeInit(using: GenerationContext) -> String {
        ""
    }
    
    func contributeImplementation(using: GenerationContext) -> String {
        let outputname = self.node.output[0]
        
        return "let _\(outputname) = _\(self.node.input[0]).reshaped(to: [\(shape.map(String.init).joined(separator: ", "))])\n"
    }
}
