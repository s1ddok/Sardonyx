class FlattenConverter: NodeConverter {
    let node: Onnx_NodeProto
    
    required init(node: Onnx_NodeProto) {
        self.node = node
    }
    
    func prepareData(using: GenerationContext) throws {
            
    }
    
    func contributeProperties(using: GenerationContext) -> String {
        "var layer_\(self.node.name) = Flatten<Float>()\n"
    }
    
    func contributeInit(using: GenerationContext) -> String {
        ""
    }
    
    func contributeImplementation(using: GenerationContext) -> String {
        let outputname = self.node.output[0]
        return "let _\(outputname) = self.layer_\(self.node.name)(_\(self.node.input[0]))\n"
    }
}
