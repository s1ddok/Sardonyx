#  Sardonyx 

> ❗️Warning: this is a half-weekend project in the beginning of it's development so please use it with caution and be ready to dive into sources in order to make it work

Sardonyx is a CLI that converts your *ONNX* model into *Swift code* + structured *data blob*, allowing you to easily reuse models that were created using frameworks like PyTorch or TensorFlow. It does all the nasty things for you as well, like transposing weights and packaing them into a single, convenient file, generating parsing logic for you. 

## Roadmap 

- [x] Introduce a proof-of-a-concept tool
- [x] Convert and test VGG19 for S4TF
- [x] Convert and test MobileNetV2 for S4TF
- [x] Support multiple inputs and outputs
- [x] Support both XLA and TFEager for S4TF
- [ ] Add more layer converters and it's variations (edge-cases, paddings)
- [ ] Support constant inputs for all the layers
- [ ] Normalize node/graph names to be valid Swift identifiers 
- [ ] Provide Sardonyx-as-a-library experience for users to provide their own custom converters
- [x] Support node sequence folding (like inject subsequent RELU into previous Conv node's activation)
- [ ] Support other scalar types outside of Float for S4TF 
- [ ] Introduce a Metal Performance Shaders backend 
- [ ] Generate a Swift Package instead of a `.swift` and `.data` files
- [ ] Generation customization: provide custom input names, custom access level and more

## Dependencies

I use `SwiftProtobuf` to generate 100% Swift *ONNX* bindings and official `Swift Argument Parser` for CLI.

## Build and Run

Currently I started with an `xcodeproj` setup but `Package.swift` will be introduced shortly. In order to run the tool on your own model you have to provide at least two arguments in a `Scheme` settings, i.e.: 

![image](https://i.imgur.com/NmWnKZN.png)

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

## Supported layers

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

## License
MIT
