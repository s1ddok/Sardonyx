import ArgumentParser
import Foundation
import libSardonyx

struct Options: ParsableArguments {
    @Option(help: ArgumentHelp("Target platform to generate code for", valueName: "target-platform"))
    var targetPlatform: TargetPlatform
    
    @Option(help: ArgumentHelp("Path to the input onnx model", valueName: "model-path"))
    var modelPath: String
    
    @Option(help: ArgumentHelp("Path to the folder where Swift sources and data will be saved", valueName: "output-directory"))
    var outputDirectory: String
    
    @Option(help: ArgumentHelp("Optional name to use for the model instead of the one backed in the graph", valueName: "model-name"))
    var modelName: String?
}
