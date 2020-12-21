module {
    func @main() {
        %x = constant 42 : i64
        %hp = hi.hpStore %x : i64 
        return
    }
}
