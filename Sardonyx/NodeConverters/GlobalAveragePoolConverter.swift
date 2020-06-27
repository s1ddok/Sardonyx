class GlobalAveragePoolConverter: NodeConverter {
    let node: Onnx_NodeProto
    
    required init(node: Onnx_NodeProto) {
        self.node = node
    }
    
    func prepareData(using: GenerationContext) throws {
            
    }
    
    func contributeProperties(using: GenerationContext) -> String {
        ""//var layer_\(self.node.name) = GlobalAvgPool2D<Float>()\n"
    }
    
    func contributeInit(using: GenerationContext) -> String {
        ""
    }
    
    func contributeImplementation(using: GenerationContext) -> String {
        let outputname = self.node.output[0]
        //return "let _\(outputname) = self.layer_\(self.node.name)(_\(self.node.input[0]))\n"
        return "let _\(outputname) = _\(self.node.input[0]).mean(alongAxes: [1, 2])\n"
    }
}
