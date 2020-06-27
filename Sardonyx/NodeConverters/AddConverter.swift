class AddConverter: NodeConverter {
    let node: Onnx_NodeProto
    required init(node: Onnx_NodeProto) {
        self.node = node
    }
    
    func prepareData(using: GenerationContext) throws {
            
    }
    
    func contributeProperties(using: GenerationContext) -> String {
        ""
    }
    
    func contributeInit(using: GenerationContext) -> String {
        ""
    }
    
    func contributeImplementation(using: GenerationContext) -> String {
        let outputname = self.node.output[0]
        return "let _\(outputname) = _\(self.node.input[0]) + _\(self.node.input[1])\n"
    }
}
