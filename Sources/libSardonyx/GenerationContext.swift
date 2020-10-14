import Foundation

public class GenerationContext {
    private(set) public var tensors: [String: Onnx_TensorProto]
    public var readCounts: [String: Int] = [:]
    
    private(set) var globalDataBlob = Data()
    
    public var sourceBuilder = SourceStringBuilder()
    
    public var inputShapes: [String: [Int]] = [:]
    
    var shouldConvertWeightsToFloat16 = false
    var shouldConvertConstantsToFloat16 = false
    
    init(graph: Onnx_GraphProto) {
        self.tensors = graph.initializer.reduce(into: [:]) { (res, tensor) in
            res[tensor.name] = tensor
        }
        
        for graphInput in graph.input {
            self.inputShapes[graphInput.name] = graphInput.type.tensorType.shape.dim.map { Int($0.dimValue) }
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
