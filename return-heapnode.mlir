module {
    func @main() -> !hi.hpnode {
        %x = hi.hpAlloc
        return %x : !hi.hpnode
    }
}
