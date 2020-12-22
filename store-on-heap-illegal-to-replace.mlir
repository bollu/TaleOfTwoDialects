// We cant replace `%x` because it maybe used by others (eg. `%y`)
// RUN: ./mir.out %s
module {
    func @main() {
        %x = constant -42 : i64
        %hp = hi.hpStore %x : i64 
        %y = std.addi %x, %x : i64
        return
    }
}
