// TODO: Support pads
// TODO: Support other dims
class MaxPoolConverter: NodeConverter {
    let node: Onnx_NodeProto
    
    required init(node: Onnx_NodeProto) {
        self.node = node
    }
    
    func contributeProperties(using context: GenerationContext) {
        context.sourceBuilder.add(line: "var layer_\(self.node.name): MaxPool2D<Float>")
    }
    
    func contributeInit(using context: GenerationContext) {
        context.sourceBuilder.add(line: "self.layer_\(self.node.name) = MaxPool2D<Float>(poolSize: (\(Int(self.node.attribute[0].ints[0])), \(Int(self.node.attribute[0].ints[1]))), strides: (\(Int(self.node.attribute[2].ints[0])), \(Int(self.node.attribute[2].ints[1]))))")
    }
    
    func contributeImplementation(using context: GenerationContext) {
        let outputname = self.node.output[0]
        context.sourceBuilder.add(line: "let _\(outputname) = self.layer_\(self.node.name)(_\(self.node.input[0]))")
    }
}
