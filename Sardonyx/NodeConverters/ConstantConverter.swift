class ConstantConverter: NodeConverter {
    let node: Onnx_NodeProto
    
    required init(node: Onnx_NodeProto) {
        self.node = node
    }
    
    func prepareData(using context: GenerationContext) throws {
        for output in self.node.output {
            context.register(tensor: self.node.attribute[0].t, with: output)
        }
    }
}
