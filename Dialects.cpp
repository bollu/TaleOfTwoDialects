#include "Dialects.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

// pattern matching
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

// dilect lowering
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
// https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/examples/toy/Ch6/toyc.cpp
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "taleTwoDialects"
#include "llvm/Support/Debug.h"

namespace mlir {

HiDialect::HiDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<HiDialect>()) {
    addOperations<HpAllocOp, HpStoreOp, BadLoweringOp>();
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

// === LOWERING ===
// === LOWERING ===
// === LOWERING ===
// === LOWERING ===
// === LOWERING ===

class HiTycon : public mlir::TypeConverter {
   public:
    using TypeConverter::convertType;

    HiTycon(MLIRContext *ctx) {
        addConversion([](Type type) { return type; });

        // hi.hpnode -> ptr.void
        addConversion([](HpnodeType type) -> Type {
            return VoidPtrType::get(type.getContext());
        });

        // int -> ptr.void
        addConversion([](IntegerType type) -> Type {
            return VoidPtrType::get(type.getContext());
        });

        // int -> !ptr.void
        addTargetMaterialization([&](OpBuilder &rewriter, VoidPtrType resultty,
                                     ValueRange vals,
                                     Location loc) -> Optional<Value> {
            if (vals.size() != 1 || !vals[0].getType().isa<IntegerType>()) {
                return {};
            }

            IntToPtrOp op = rewriter.create<IntToPtrOp>(loc, vals[0]);
            llvm::SmallPtrSet<Operation *, 1> exceptions;
            exceptions.insert(op);

            // vvv HACK/MLIRBUG: isn't this a hack? why do I need this?
            // vals[0].replaceAllUsesExcept(op.getResult(), exceptions);
            return op.getResult();
        });
    }
};

class HpStoreConversionPattern : public ConversionPattern {
   public:
    explicit HpStoreConversionPattern(TypeConverter &tc, MLIRContext *context)
        : ConversionPattern(HpStoreOp::getOperationName(), 1, tc, context) {}

    // mkConstructor: !ptr.void -> !ptr.void
    static FuncOp getOrInsertHpStore(PatternRewriter &rewriter,
                                     ModuleOp module) {
        const std::string name = "hpstore_runtime";
        if (FuncOp fn = module.lookupSymbol<FuncOp>(name)) {
            return fn;
        }

        SmallVector<mlir::Type, 4> argTys = {
            VoidPtrType::get(rewriter.getContext())};
        mlir::Type retty = VoidPtrType::get(rewriter.getContext());

        auto fntype = rewriter.getFunctionType(argTys, retty);
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());

        FuncOp fndecl =
            rewriter.create<FuncOp>(rewriter.getUnknownLoc(), name, fntype);

        fndecl.setPrivate();
        return fndecl;
    }

    LogicalResult matchAndRewrite(
        Operation *op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        HpStoreOp store = cast<HpStoreOp>(op);

        FuncOp fn =
            getOrInsertHpStore(rewriter, op->getParentOfType<ModuleOp>());
        rewriter.setInsertionPointAfter(store);
        rewriter.replaceOpWithNewOp<CallOp>(store, fn, store.getOperand());

        llvm::errs() << "\n====\n";
        fn.getParentOp()->dump();
        llvm::errs() << "\n===\n";
        getchar();

        return success();
    }
};

struct LowerHiPass : public Pass {
    LowerHiPass() : Pass(mlir::TypeID::get<LowerHiPass>()){};
    StringRef getName() const override { return "LowerHiPass"; }

    std::unique_ptr<Pass> clonePass() const override {
        auto newInst = std::make_unique<LowerHiPass>(
            *static_cast<const LowerHiPass *>(this));
        newInst->copyOptionValuesFrom(this);
        return newInst;
    }

    void runOnOperation() override {
        assert(isa<ModuleOp>(getOperation()));
        
        ConversionTarget target(getContext());
        target.addIllegalDialect<HiDialect>();
        target.addLegalDialect<StandardOpsDialect>();
        target.addLegalDialect<PtrDialect>();
        target.addLegalDialect<scf::SCFDialect>();
        target.addLegalOp<ModuleOp, ModuleTerminatorOp, FuncOp>();
        ::llvm::DebugFlag = true;

        HiTycon tycon(&getContext());

        // applyPartialConversion | applyFullConversion
        mlir::OwningRewritePatternList patterns;
        patterns.insert<HpStoreConversionPattern>(tycon, &getContext());

        if (failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
            llvm::errs() << "\n===Hi lowering failed at Conversion===\n";
            getOperation()->print(llvm::errs());
            llvm::errs() << "\n===\n";
            signalPassFailure();
        };

        llvm::errs() << "\n===Module after conversion, before vefication: ===\n";
        getOperation()->dump();

        llvm::errs() << "\n===Verifying lowering...===\n";
        // if (failed(mlir::verify(getOperation()))) {
        //     llvm::errs() << "===Hi lowering failed at Verification===\n";
        //     getOperation()->print(llvm::errs());
        //     llvm::errs() << "\n===\n";
        //     signalPassFailure();
        // }

        ::llvm::DebugFlag = false;
    };
};

std::unique_ptr<mlir::Pass> createLowerHiPass() {
    return std::make_unique<LowerHiPass>();
}

void registerLowerHiPass() {
    ::mlir::registerPass(
        "hi-lower", "lower hi",
        []() -> std::unique_ptr<::mlir::Pass> { return createLowerHiPass(); });
}

}  // namespace mlir
