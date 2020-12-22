// RUN: ./mir.out %s
module {
    func @main_use_heapnode(%x: !hi.hpnode) {
        return
    }
}

