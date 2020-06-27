protocol NodeConverter {
    init(node: Onnx_NodeProto)
    func prepareData(using: GenerationContext) throws
    func contributeProperties(using: GenerationContext) -> String
    func contributeInit(using: GenerationContext) -> String
    func contributeImplementation(using: GenerationContext) -> String
}
