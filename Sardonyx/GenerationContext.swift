import Foundation

class GenerationContext {
    let tensors: [String: Onnx_TensorProto]
    
    private(set) var globalDataBlob = Data()
    
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
}
