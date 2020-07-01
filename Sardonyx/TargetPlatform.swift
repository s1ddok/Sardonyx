import ArgumentParser

enum TargetPlatform: String {
    case s4tf
    case mps
}

extension TargetPlatform: ExpressibleByArgument {
    
}
