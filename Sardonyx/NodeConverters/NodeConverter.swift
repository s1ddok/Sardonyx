protocol NodeConverter {
    init(node: Onnx_NodeProto)
    func prepareData(using: GenerationContext) throws
    func contributeProperties(using: GenerationContext)
    func contributeInit(using: GenerationContext)
    func contributeImplementation(using: GenerationContext)
}

extension NodeConverter {
    func prepareData(using: GenerationContext) throws {}
    func contributeProperties(using: GenerationContext) {}
    func contributeInit(using: GenerationContext) {}
    func contributeImplementation(using: GenerationContext) {}
}
