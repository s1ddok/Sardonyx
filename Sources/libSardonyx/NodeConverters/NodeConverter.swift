protocol NodeConverter {
    init(node: Onnx_NodeProto)
    // this array should represent list of inputs that are ideologically equal to graph nodes
    // e.g. for Upsample layers: first input should be considered as a graph input, but second input is not
    // because it is only used to configure the layer
    var graphInputs: [String] { get }
    var graphOutputs: [String] { get }
    func prepareData(using: GenerationContext) throws
    func contributeProperties(using: GenerationContext)
    func contributeInit(using: GenerationContext)
    func contributeImplementation(using: GenerationContext)
}

extension NodeConverter {
    var graphInputs: [String] { [] }
    var graphOutputs: [String] { [] }
    func prepareData(using: GenerationContext) throws {}
    func contributeProperties(using: GenerationContext) {}
    func contributeInit(using: GenerationContext) {}
    func contributeImplementation(using: GenerationContext) {}
}
