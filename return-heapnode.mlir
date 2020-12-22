// Here, we try to return a heapnode. We expect the type converter to convert
// this into a pointer type. We know that this works from the @main_return_void.

// However, the lowering does not automatically insert the hpnod2ptr for
// whatever reason. Or rather, it inserts it, and then proceeds to ignore it.
module {
    func @main() -> !hi.hpnode {
        %x = hi.hpAlloc
        return %x : !hi.hpnode
    }

    // func @main_return_void() -> !ptr.void {
    //     %x = hi.hpAlloc
    //     %y = ptr.hpnode2ptr %x
    //     return %y : !ptr.void
    // }
}
