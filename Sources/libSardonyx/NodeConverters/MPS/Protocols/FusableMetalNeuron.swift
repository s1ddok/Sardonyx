protocol FusableMetalNeuron {
    var descriptor: String { get }
    var neuron: String { get }
    var output: String { get }
}

protocol FusableMetalNeuronInjectable: AnyObject {
    var neuron: FusableMetalNeuron? { get set }
}
