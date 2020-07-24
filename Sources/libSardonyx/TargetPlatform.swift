import ArgumentParser

public enum TargetPlatform: String {
    case s4tf
    case mps
}

extension TargetPlatform: ExpressibleByArgument {
    
}
