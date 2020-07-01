class PadMetalConverter: NodeConverter {
    let node: Onnx_NodeProto
    
    var bPadding: (Int, Int)
    var cPadding: (Int, Int)
    var yPadding: (Int, Int)
    var xPadding: (Int, Int)
    var value: Float?
    var mode: String
    
    required init(node: Onnx_NodeProto) {
        self.node = node
        
        self.mode = String(data: node.attribute[0].s, encoding: .utf8)!
        let pads = self.node.attribute[1].ints.map(Int.init)
        
        self.bPadding = (pads[0], pads[4])
        self.cPadding = (pads[1], pads[5])
        self.yPadding = (pads[2], pads[6])
        self.xPadding = (pads[3], pads[7])
        
        self.value = self.node.attribute[safe: 2]?.f
        assert(self.mode != "edge", "Edge padding mode isn't supported by Swift for TensorFlow")
    }
    
    
}
