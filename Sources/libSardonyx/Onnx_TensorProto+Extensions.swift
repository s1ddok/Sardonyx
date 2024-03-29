import Foundation

extension Onnx_TensorProto {

    public var integers: [Int] {
        switch Int(self.dataType) {
        case DataType.int32.rawValue,
             DataType.int16.rawValue,
             DataType.int8.rawValue,
             DataType.uint16.rawValue,
             DataType.uint8.rawValue,
             DataType.bool.rawValue:
            return self.int32Data.map(Int.init)
        case DataType.int64.rawValue:
            if self.int64Data.count == 0 {
                return self.rawData.withUnsafeBytes { (p: UnsafeRawBufferPointer) in
                    return p.bindMemory(to: Int64.self)
                }.map(Int.init)
            }
            
            return self.int64Data.map(Int.init)
        case DataType.uint32.rawValue,
             DataType.uint64.rawValue:
            return self.uint64Data.map(Int.init)
        case DataType.float.rawValue:
            if self.floatData.count == 0 {
                return self.rawData.withUnsafeBytes { (p: UnsafeRawBufferPointer) in
                    return p.bindMemory(to: Float.self)
                }.map(Int.init)
            }
            return self.floatData.map(Int.init)
        case DataType.double.rawValue:
            return self.doubleData.map(Int.init)
//        case DataType.float16.rawValue:
//            self.rawData.withUnsafeBytes { (p: UnsafeRawBufferPointer) in
//                if #available(OSX 9999, *) {
//                    return p.bindMemory(to: Float16.self)
//                } else {
//                    fatalError()
//                }
//            }.map(Int.init)
        default:
            fatalError("Unsupported conversion rule")
        }
    }

    public var floats: [Float] {
        switch Int(self.dataType) {
        case DataType.int32.rawValue,
             DataType.int16.rawValue,
             DataType.int8.rawValue,
             DataType.uint16.rawValue,
             DataType.uint8.rawValue,
             DataType.bool.rawValue:
            return self.int32Data.map(Float.init)
        case DataType.int64.rawValue:
            return self.int64Data.map(Float.init)
        case DataType.uint32.rawValue,
             DataType.uint64.rawValue:
            return self.uint64Data.map(Float.init)
        case DataType.float.rawValue:
            if self.floatData.count == 0 {
                return self.rawData.withUnsafeBytes { (p: UnsafeRawBufferPointer) in
                    Array(p.bindMemory(to: Float.self))
                }
            }
            return self.floatData
        case DataType.double.rawValue:
            return self.doubleData.map(Float.init)
//        case DataType.float16.rawValue:
//            let count = self.rawData.count / MemoryLayout<Float16>.stride
//            return self.rawData.withUnsafeBytes {
//                float16to32(UnsafeMutableRawPointer(mutating: $0),
//                            count: count)
//            } ?? []
        default:
             fatalError("Unsupported conversion rule")
        }
    }

    public var length: Int {
        return Int(self.dims.reduce(1, *))
    }

}
