protocol ActivationConverter {
    var closure: String { get }
    
    var output: String { get }
}

protocol ActivationConverterInjectable: AnyObject {
    var activationConverter: ActivationConverter? { get set }
}
