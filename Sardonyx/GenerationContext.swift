import Foundation

class GenerationContext {
    private(set) var tensors: [String: Onnx_TensorProto]
    
    private(set) var globalDataBlob = Data()
    
    public var sourceBuilder = SourceStringBuilder()
    
    init(graph: Onnx_GraphProto) {
        self.tensors = graph.initializer.reduce(into: [:]) { (res, tensor) in
            res[tensor.name] = tensor
        }
    }
    
    func add(data: Data) -> Int {
        let offset = self.globalDataBlob.count
        
        self.globalDataBlob.append(data)
        
        return offset
    }
    
    func register(tensor: Onnx_TensorProto, with name: String) {
        self.tensors[name] = tensor
    }
}
