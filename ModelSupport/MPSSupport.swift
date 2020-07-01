import Metal
import MetalPerformanceShaders

public typealias Kernel = (height: Int, width: Int)
public typealias Strides = (height: Int, width: Int)
public typealias Dilations = (height: Int, width: Int)
public typealias Pads = (top: Int, left: Int, bottom: Int, right: Int)
public typealias Padding = (height: Int, width: Int)
public typealias Scales = (height: Int, width: Int)

@objc(ONNXConvolutionPadding) public class ONNXConvolutionPadding: NSObject, MPSNNPadding {
    public let kernel: Kernel
    public let dilations: Dilations
    public let strides: Strides
    public let pads: Pads
    public let outputPadding: Padding
    public let isTranspose: Bool

    public init(kernel: Kernel,
                strides: Strides,
                dilations: Dilations,
                pads: Pads,
                outputPadding: Padding,
                isTranspose: Bool) {
        self.kernel = kernel
        self.dilations = dilations
        self.strides = strides
        self.pads = pads
        self.outputPadding = outputPadding
        self.isTranspose = isTranspose
    }

    required convenience public init?(coder aDecoder: NSCoder) {
        guard
            let data = aDecoder.decodeData(),
            let other = NSKeyedUnarchiver.unarchiveObject(with: data) as? ONNXConvolutionPadding
        else { return nil }
        self.init(kernel: other.kernel,
                  strides: other.strides,
                  dilations: other.dilations,
                  pads: other.pads,
                  outputPadding: other.outputPadding,
                  isTranspose: other.isTranspose)
    }

    public func encode(with aCoder: NSCoder) {
        aCoder.encode(NSKeyedArchiver.archivedData(withRootObject: self))
    }

    public func paddingMethod() -> MPSNNPaddingMethod {
        return [.custom]
    }

    public func destinationImageDescriptor(forSourceImages sourceImages: [MPSImage],
                                           sourceStates: [MPSState]?,
                                           for kernel: MPSKernel,
                                           suggestedDescriptor inDescriptor: MPSImageDescriptor) -> MPSImageDescriptor {
        let inputHeight = sourceImages[0].height
        let inputWidth = sourceImages[0].width

        if self.isTranspose {
            let conv = kernel as! MPSCNNConvolutionTranspose
            conv.offset = MPSOffset(x: 0, y: 0, z: 0)
            conv.edgeMode = .zero
            conv.kernelOffsetX = self.kernel.width / 2 - self.kernel.width + 1 + self.pads.left
            conv.kernelOffsetY = self.kernel.height / 2 - self.kernel.height + 1 + self.pads.top
        } else {
            let conv = kernel as! MPSCNNConvolution
            conv.offset = MPSOffset(x: self.kernel.width / 2 - self.pads.left,
                                    y: self.kernel.height / 2 - self.pads.top,
                                    z: 0)
            conv.edgeMode = .zero
        }
        let paddedSize = self.paddedSize(inputWidth: inputWidth,
                                         inputHeight: inputHeight)
        inDescriptor.height = paddedSize.height
        inDescriptor.width = paddedSize.width

        return inDescriptor
    }

    public func paddedSize(inputWidth: Int,
                           inputHeight: Int) -> (width: Int, height: Int) {
        let height: Int
        let width: Int
        if self.isTranspose {
            height = (inputHeight - 1) * self.strides.height
                - self.pads.top - self.pads.bottom
                + self.kernel.height + self.outputPadding.height
            width = (inputWidth - 1) * self.strides.width
                - self.pads.left - self.pads.right
                + self.kernel.width + self.outputPadding.width
        } else {
            height = (inputHeight + self.pads.top
                + self.pads.bottom - self.kernel.height)
                / self.strides.height + 1
            width = (inputWidth + self.pads.left
                + self.pads.right - self.kernel.width)
                / self.strides.width + 1
        }
        return (width, height)
    }

    public static var supportsSecureCoding: Bool = true
}


@objc class ConvolutionDataSource: NSObject, MPSCNNConvolutionDataSource {
    let _descriptor: MPSCNNConvolutionDescriptor
    let _weights: UnsafeMutableRawPointer
    let _bias: UnsafeMutablePointer<Float>?
    
    init(weights: UnsafeMutableRawPointer,
         bias: UnsafeMutablePointer<Float>?,
         outputChannels: Int,
         kernelHeight: Int,
         kernelWidth: Int,
         inputChannels: Int,
         dilations: (Int, Int),
         strides: (Int, Int),
         groups: Int,
         neuron: MPSCNNNeuron?) {
        let isDepthwise = (groups != 1) && (groups == outputChannels)
        
        if isDepthwise {
            _descriptor = MPSCNNDepthWiseConvolutionDescriptor(
                kernelWidth: kernelWidth,
                kernelHeight: kernelHeight,
                inputFeatureChannels: outputChannels,
                outputFeatureChannels: outputChannels,
                neuronFilter: neuron)
            _descriptor.groups = 1
        } else {
            _descriptor = MPSCNNConvolutionDescriptor(
                kernelWidth: kernelWidth,
                kernelHeight: kernelHeight,
                inputFeatureChannels: inputChannels,
                outputFeatureChannels: outputChannels,
                neuronFilter: neuron)
            _descriptor.groups = groups
        }

        _descriptor.dilationRateY = dilations.0
        _descriptor.dilationRateX = dilations.1
        _descriptor.strideInPixelsY = strides.0
        _descriptor.strideInPixelsX = strides.1
        
        //_descriptor.setBatchNormalizationParametersForInferenceWithMean(<#T##mean: UnsafePointer<Float>?##UnsafePointer<Float>?#>, variance: <#T##UnsafePointer<Float>?#>, gamma: <#T##UnsafePointer<Float>?#>, beta: <#T##UnsafePointer<Float>?#>, epsilon: <#T##Float#>)
        self._weights = weights
        self._bias = bias
    }
    
    func copy(with zone: NSZone? = nil) -> Any {
        return self.mutableCopy()
    }

    func dataType() -> MPSDataType {
        return .float32
    }

    @available(OSX 10.13, *)
    func descriptor() -> MPSCNNConvolutionDescriptor {
        return self._descriptor
    }

    func weights() -> UnsafeMutableRawPointer {
        return self._weights
    }

    func biasTerms() -> UnsafeMutablePointer<Float>? {
        return self._bias
    }

    func load() -> Bool {
        return true
    }

    func purge() {}

    func label() -> String? {
        return "Convolution data source"
    }
}

@objc class InstanceNormDataSource: NSObject, MPSCNNInstanceNormalizationDataSource {
    private(set) var numberOfFeatureChannels: Int
    private(set) var gammas: UnsafeMutablePointer<Float>?
    private(set) var betas: UnsafeMutablePointer<Float>?
    
    public init(channels: Int, gammas: UnsafeMutablePointer<Float>?, betas: UnsafeMutablePointer<Float>?) {
        self.numberOfFeatureChannels = channels
        self.gammas = gammas
        self.betas = betas
    }
    
    func gamma() -> UnsafeMutablePointer<Float>? {
        return self.gammas
    }

    func beta() -> UnsafeMutablePointer<Float>? {
        return self.betas
    }

    func label() -> String? {
        return "Instance norm data source"
    }

    required init?(coder aDecoder: NSCoder) {
        fatalError()
    }

    func copy(with zone: NSZone? = nil) -> Any {
        return self.mutableCopy()
    }

}
