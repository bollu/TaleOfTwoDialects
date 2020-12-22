// Check that we can lower returns of arguments correctly.
// RUN: ./mir.out %s
module {
    func @main_use_heapnode(%x: !hi.hpnode) -> !hi.hpnode{
        return %x : !hi.hpnode
    }
}

