#pragma once
#include <llvm/ADT/ArrayRef.h>

#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"

// implementation stuff
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

class HiDialect : public mlir::Dialect {
   public:
    explicit HiDialect(mlir::MLIRContext *context);
    static llvm::StringRef getDialectNamespace() { return "hi"; }
    mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
    void printType(mlir::Type type,
                   mlir::DialectAsmPrinter &printer) const override;
};

class HiType : public Type {
   public:
    using Type::Type;

    using Type::getAsOpaquePointer;
    static HiType getFromOpaquePointer(const void *ptr) {
        return HiType(static_cast<ImplType *>(const_cast<void *>(ptr)));
    }
    static bool classof(Type type) {
        return llvm::isa<HiDialect>(type.getDialect());
    }
    HiDialect &getDialect() {
        return static_cast<HiDialect &>(Type::getDialect());
    }
};

class HpnodeType
    : public mlir::Type::TypeBase<HpnodeType, HiType, TypeStorage> {
   public:
    using Base::Base;
    static HpnodeType get(MLIRContext *context) { return Base::get(context); }
};

class HpAllocOp : public Op<HpAllocOp, OpTrait::OneResult> {
   public:
    using Op::Op;
    static StringRef getOperationName() { return "hi.hpAlloc"; };
    void print(OpAsmPrinter &p) {
        return p.printGenericOp(this->getOperation());
    }

    static ParseResult parse(OpAsmParser &parser, OperationState &result) {
        result.addTypes(HpnodeType::get(parser.getBuilder().getContext()));
        return success();
    };
};

class HpStoreOp : public Op<HpStoreOp, OpTrait::OneResult> {
   public:
    using Op::Op;
    static StringRef getOperationName() { return "hi.hpStore"; };
    void print(OpAsmPrinter &p) {
        return p.printGenericOp(this->getOperation());
    }
    static ParseResult parse(OpAsmParser &parser, OperationState &result) {
        OpAsmParser::OperandType rand;  // ope'rand
        Type ty;
        if (parser.parseOperand(rand) || parser.parseColonType(ty) ||
            parser.resolveOperand(rand, ty, result.operands)) {
            return failure();
        }
        result.addTypes(HpnodeType::get(parser.getBuilder().getContext()));
        return success();
    };
};


// === PTR DIALECT ===
// === PTR DIALECT ===
// === PTR DIALECT ===

class PtrDialect : public mlir::Dialect {
   public:
    explicit PtrDialect(mlir::MLIRContext *context);
    static llvm::StringRef getDialectNamespace() { return "ptr"; }
    mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
    void printType(mlir::Type type,
                   mlir::DialectAsmPrinter &printer) const override;
};

class PtrType : public Type {
   public:
    using Type::Type;

    using Type::getAsOpaquePointer;
    static PtrType getFromOpaquePointer(const void *ptr) {
        return PtrType(static_cast<ImplType *>(const_cast<void *>(ptr)));
    }
    static bool classof(Type type) {
        return llvm::isa<PtrDialect>(type.getDialect());
    }
    PtrDialect &getDialect() {
        return static_cast<PtrDialect &>(Type::getDialect());
    }
};

class VoidPtrType
    : public mlir::Type::TypeBase<VoidPtrType, PtrType, TypeStorage> {
   public:
    using Base::Base;
    static VoidPtrType get(MLIRContext *context) { return Base::get(context); }
};

// %ptr = inttoptr %i
class IntToPtrOp
    : public Op<IntToPtrOp, OpTrait::OneResult, OpTrait::OneOperand> {
   public:
    using Op::Op;
    static StringRef getOperationName() { return "ptr.inttoptr"; };
    static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      Value vint) {
        assert(vint.getType().isa<IntegerType>());
        state.addOperands(vint);
        state.addTypes(VoidPtrType::get(builder.getContext()));
    }
    void print(OpAsmPrinter &p) {
        return p.printGenericOp(this->getOperation());
    }
};


}  // namespace mlir
