#  Sardonyx 

> ❗️Warning: this project is in the beginning of its' development so please use it with caution and be ready to dive into sources in order to make it work

Sardonyx is a CLI that converts your *ONNX* model into *Swift code* + structured *data blob*, allowing you to easily reuse models that were created using frameworks like PyTorch or TensorFlow. It does all the nasty things for you as well, like transposing weights and packing them into a single, convenient file, while also generating parsing logic for you. 

## Supported target platforms
- Swift for TensorFlow
- Metal Performance Shaders 

Adding BNNS, MLCompute and code for other languages is under consideration, but getting first two to work on a wide variate of models is a first priority for me. 

## Roadmap 

- [x] Introduce a proof-of-a-concept tool
- [x] Convert and test VGG19 for S4TF
- [x] Convert and test MobileNetV2 for S4TF
- [x] Support multiple inputs and outputs
- [x] Support both XLA and TFEager for S4TF
- [ ] Add more layer converters and it's variations (edge-cases, paddings)
- [x] Support constant inputs for all the layers
- [ ] Normalize node/graph names to be valid Swift identifiers 
- [ ] Provide Sardonyx-as-a-library experience for users to provide their own custom converters
- [x] Support node sequence folding (like inject subsequent RELU into previous Conv node's activation)
- [ ] Support other scalar types outside of Float (i.e. Float16 and quantized weights)
- [x] Introduce a Metal Performance Shaders backend 
- [ ] Generate a Swift Package instead of a `.swift` and `.data` files
- [ ] Generation customization: provide custom input names, custom access level, additional outputs and more

## Dependencies

I use `SwiftProtobuf` to generate 100% Swift *ONNX* bindings and official `Swift Argument Parser` for CLI. Some generated models require help classes to run, I provide them in this repo. 

## Build and Run

You can run Sardonyx as a CLI with SwiftPM, or use it as a library (advanced usage for custom converters provision):

```
swift run -c release Sardonyx --target-platform=s4tf --model-path=vgg19-7.onnx --output-directory=.
```

You have to specify target platform by `--target-platform` option. Currently it acceps `s4tf` or `mps` value.

### S4TF

I tested this tool on [VGG-16](https://github.com/onnx/models/blob/master/vision/classification/vgg/model/vgg16-7.onnx), [VGG-19](https://github.com/onnx/models/blob/master/vision/classification/vgg/model/vgg19-7.onnx) and [MobileNetV2](https://github.com/onnx/models/blob/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx) from official ONNX model-zoo. Please don't expect it to work on anything else for now. You check support layer list to see whether or not you can convert your model now

If everything goes fine you will find two files in your folder: `<YOURMODEL>.swift` and `<YOURMODEL>.data`. Please add them into your Swift 4 TensorFlow project. Once you do this, usage is pretty straightforward, this is a MobileNetV2 example:


```swift
import TensorFlow
import Foundation
import ModelSupport

let data = try! Data(contentsOf: URL(fileURLWithPath: "/**/*/<YOURMODEL>.data"))
let model = data.withUnsafeBytes { p -> main in
    main(data: p.baseAddress!, device: .defaultXLA)
}

Context.local.learningPhase = .inference
let kittenURL =  URL(fileURLWithPath: "/**/*/kitten/jpeg")
let image = Image(jpeg: kittenURL).resized(to: (224, 224)).tensor.expandingShape(at: 0)

let mean = Tensor<Float>([0.485, 0.456, 0.406], on: .defaultXLA)
let std = Tensor<Float>([0.229, 0.224, 0.225], on: .defaultXLA)
let xlaImage = Tensor<Float>.init(copying: image, to: .defaultXLA)
let input = (xlaImage / 255.0 - mean) / std
let output = model(input)

let smo = softmax(output)
print(smo.max())
```

### MPS

Metal supported is limited compared to S4TF, but I am working on this. Minimal usage example (based on [Alloy](https://github.com/s1ddok/Alloy) syntax) is here:

```swift
let model = torch_jit_export(data: pointer.baseAddress!, device: context.device)
let result = try! context.scheduleAndWait { buffer -> MPSImage in
    imageScale.encode(commandBuffer: buffer, sourceTexture: texture, destinationTexture: scaledTexture)
    let mpsimage = MPSImage(texture: scaledTexture, featureChannels: 3)
    return model.encode(commandBuffer: buffer, _input_1: mpsimage)
}
```

## Supported layers

### S4TF
- Conv2D
- MaxPool2D
- Flatten
- Reshape
- BatchNormalization
- Gemm
- GlobalAveragePool2D
- Same/Valid paddings
- Dropout
- Add
- RELU
- Constant
- ConvTranspose2D
- InstanceNorm
- Pad
- Softmax
- Sigmoid
- LeakyRelu
- Tanh
- Upsample
- Mul

### MPS
- Pad
- Conv2D
- InstanceNormalization
- ConvTranspose2D
- Add
- RELU
- Sigmoid

## License
MIT
