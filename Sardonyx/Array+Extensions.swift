extension Array {
    public func transposed(to order: [Int], assuming shape: [Int]) -> [Element] where Element: Numeric {
        precondition(self.count == shape.reduce(1, *))
        let ndim = order.count
        
        let newShape = shape.indices.map { shape[order[$0]] }
        var newArray = [Element].init(repeating: 0, count: newShape.reduce(1, *))
        
        let srcShape = shape
        let dstStride = newShape.enumerated().map { idx, dim -> Int in
            if idx == newShape.count - 1 { return 1 }
            
            return newShape.dropFirst(idx + 1).reduce(1, *)
        }
        var idx = [Int](repeating: 0, count: ndim)

        for j in 0..<count {
            // Map the source index to the destination index.
            var dstIndex = 0
            for i in 0..<ndim {
                dstIndex += idx[order[i]] * dstStride[i]
            }
            
            // Copy the value.
            newArray[dstIndex] = self[j]
            
            // Update the source index.
            var i = ndim - 1
            idx[i] += 1
            while i > 0 && idx[i] >= srcShape[i] {
                idx[i] = 0
                idx[i - 1] += 1
                i -= 1
            }
        }
        return newArray
    }
    
    public func reformatConvWeightToMPS(outputChannels: Int,
                                        inputChannels: Int,
                                        kernelHeight: Int,
                                        kernelWidth: Int,
                                        isTranspose: Bool) -> [Element] {
        var data: [Element] = [Element](repeating: self[0], count: self.count)
        for oc in 0..<outputChannels {
            for ic in 0..<inputChannels {
                for kh in 0..<kernelHeight {
                    for kw in 0..<kernelWidth {
                        let inputIdx: Int
                        let outputIdx: Int
                        
                        if isTranspose {
                            inputIdx = ic * outputChannels * kernelHeight * kernelWidth
                                + oc * kernelHeight * kernelWidth
                                + kh * kernelWidth + kw
                            outputIdx = oc * kernelHeight * kernelWidth * inputChannels
                                + (kernelHeight - 1 - kh) * kernelWidth * inputChannels
                                + (kernelWidth - 1 - kw) * inputChannels + ic
                        } else {
                            inputIdx = oc * inputChannels * kernelHeight * kernelWidth
                                + ic * kernelHeight * kernelWidth
                                + kh * kernelWidth + kw
                            outputIdx = oc * inputChannels * kernelHeight * kernelWidth
                                + kh * kernelWidth * inputChannels
                                + kw * inputChannels + ic
                        }
                        
                        data[outputIdx] = self[inputIdx]
                    }
                }
            }
        }
        
        return data
    }
    
    /// Returns the element at the specified index if it is within bounds, otherwise nil.
    internal subscript (safe index: Index) -> Element? {
        return self.indices.contains(index) ? self[index] : nil
    }
}
