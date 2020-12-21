// make this crash even if we replace the `constant 42` by adding an operation
// that *incorrectly lowers*. This causes MLIR to go into a bad state where
// we have replaced the old value, but cant recover from failure.
module {
    func @main() {
        %x = constant 42 : i64
        %hp = hi.hpStore %x : i64 
        hi.badLowering
        return
    }
}
