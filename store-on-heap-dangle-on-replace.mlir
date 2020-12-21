// If we replace the `int` in `int2ptr: int -> void` type conversion pattern,
// we can create dangling IR. In this care, the argument for hpStore will be
// <<UNKNOWN SSA VALUE>> since we are forced to back out of the rewrite on
// encountering the `hi.noLowering` AFTER having already overwritten the use `%x`.
module {
    func @main() {
        %x = constant 42 : i64
        %hp = hi.hpStore %x : i64 
        hi.noLowering
        return
    }
}
