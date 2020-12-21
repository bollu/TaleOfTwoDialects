#include "Dialects.h"

namespace mlir {

HiDialect::HiDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<HiDialect>()) {
    addOperations<HpAllocOp, HpStoreOp>();
    addTypes<HpnodeType>();
}

mlir::Type HiDialect::parseType(mlir::DialectAsmParser &p) const {
    if (succeeded(p.parseOptionalKeyword("hpnode"))) {
        return HpnodeType::get(p.getBuilder().getContext());
    }

    assert(false && "unknown type in high dialect");
}

void HiDialect::printType(mlir::Type t, mlir::DialectAsmPrinter &p) const {
    if (t.isa<HpnodeType>()) {
        p << "hpnode";
        return;
    }
    assert(false && "unknown type");
}

// === PTR DIALECT ===
// === PTR DIALECT ===
// === PTR DIALECT ===
// === PTR DIALECT ===
// === PTR DIALECT ===
PtrDialect::PtrDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<PtrDialect>()) {
    addOperations<IntToPtrOp>();
    addTypes<VoidPtrType>();
}

mlir::Type PtrDialect::parseType(mlir::DialectAsmParser &p) const {
    if (succeeded(p.parseOptionalKeyword("void"))) {
        return VoidPtrType::get(p.getBuilder().getContext());
    }

    assert(false && "unknown type in high dialect");
}

void PtrDialect::printType(mlir::Type t, mlir::DialectAsmPrinter &p) const {
    if (t.isa<VoidPtrType>()) {
        p << "void";
        return;
    }
    assert(false && "unknown type");
}

}  // namespace mlir
