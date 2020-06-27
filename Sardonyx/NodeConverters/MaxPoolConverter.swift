// TODO: Support pads
// TODO: Support other dims
class MaxPoolConverter: NodeConverter {
    let node: Onnx_NodeProto
    
    required init(node: Onnx_NodeProto) {
        self.node = node
    }
    
    func prepareData(using: GenerationContext) throws {
            
    }
    
    func contributeProperties(using: GenerationContext) -> String {
        "var layer_\(self.node.name): MaxPool2D<Float>\n"
    }
    
    func contributeInit(using: GenerationContext) -> String {
        "self.layer_\(self.node.name) = MaxPool2D<Float>(poolSize: (\(Int(self.node.attribute[0].ints[0])), \(Int(self.node.attribute[0].ints[1]))), strides: (\(Int(self.node.attribute[2].ints[0])), \(Int(self.node.attribute[2].ints[1]))))\n"
    }
    
    func contributeImplementation(using: GenerationContext) -> String {
        let outputname = self.node.output[0]
        return "let _\(outputname) = self.layer_\(self.node.name)(_\(self.node.input[0]))\n"
    }
}
