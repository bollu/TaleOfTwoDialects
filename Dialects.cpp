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
    addOperations<IntToPtrOp, HpnodeToPtrOp>();
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

        // hi.hpnode -> !ptr.void
        addTargetMaterialization([&](OpBuilder &rewriter, VoidPtrType resultty,
                                     ValueRange vals,
                                     Location loc) -> Optional<Value> {
            if (vals.size() != 1 || !vals[0].getType().isa<IntegerType>()) {
                return {};
            }

            HpnodeToPtrOp op = rewriter.create<HpnodeToPtrOp>(loc, vals[0]);
            llvm::SmallPtrSet<Operation *, 1> exceptions;
            exceptions.insert(op);

            // vvv HACK/MLIRBUG: isn't this a hack? why do I need this?
            // vals[0].replaceAllUsesExcept(op.getResult(), exceptions);
            return op.getResult();
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

// https://github.com/spcl/open-earth-compiler/blob/master/lib/Conversion/StencilToStandard/ConvertStencilToStandard.cpp#L45
class FuncOpLowering : public ConversionPattern {
   public:
    explicit FuncOpLowering(TypeConverter &tc, MLIRContext *context)
        : ConversionPattern(FuncOp::getOperationName(), 1, tc, context) {}

    LogicalResult matchAndRewrite(
        Operation *operation, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto funcOp = cast<FuncOp>(operation);

        TypeConverter::SignatureConversion inputs(funcOp.getNumArguments());
        for (auto &en : llvm::enumerate(funcOp.getType().getInputs()))
            inputs.addInputs(en.index(),
                             typeConverter->convertType(en.value()));

        TypeConverter::SignatureConversion results(funcOp.getNumResults());
        for (auto &en : llvm::enumerate(funcOp.getType().getResults()))
            results.addInputs(en.index(),
                              typeConverter->convertType(en.value()));

        auto funcType =
            FunctionType::get(inputs.getConvertedTypes(),
                              results.getConvertedTypes(), funcOp.getContext());

        // Replace the function by a function with an updated signature
        auto newFuncOp = rewriter.create<FuncOp>(loc, funcOp.getName(),
                                                 funcType, llvm::None);
        rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                    newFuncOp.end());

        // Convert the signature and delete the original operation
        rewriter.applySignatureConversion(&newFuncOp.getBody(), results);
        rewriter.eraseOp(funcOp);
        return success();
    }
};

class ReturnOpLowering : public ConversionPattern {
   public:
    explicit ReturnOpLowering(TypeConverter &tc, MLIRContext *context)
        : ConversionPattern(ReturnOp::getOperationName(), 1, tc, context) {}

    LogicalResult matchAndRewrite(
        Operation *operation, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto retOp = cast<ReturnOp>(operation);

        // TypeConverter::SignatureConversion inputs(retOp.getNumOperands());
        // for (auto &en : llvm::enumerate(retOp.getOperandTypes()))
        //     inputs.addInputs(en.index(),
        //                      typeConverter->convertType(en.value()));

        // Replace the function by a function with an updated signature
        auto newRetOp = rewriter.create<ReturnOp>(loc, operands);
        rewriter.eraseOp(operation);
        // rewriter.replaceOp(operation, newRetOp);
        return success();
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

        llvm::errs() << "\nvvvvHpStoreConversionPattern beforevvvv\n";
        fn.getParentOp()->dump();

        rewriter.setInsertionPointAfter(store);
        CallOp call = rewriter.create<CallOp>(store.getLoc(), fn, operands);
        rewriter.replaceOp(store, call.getResults());

        llvm::errs() << "\n===after===\n";
        fn.getParentOp()->dump();
        llvm::errs() << "\n^^^^\n";
        getchar();

        return success();
    }
};

class HpAllocOpConversionPattern : public ConversionPattern {
   public:
    explicit HpAllocOpConversionPattern(TypeConverter &tc, MLIRContext *context)
        : ConversionPattern(HpAllocOp::getOperationName(), 1, tc, context) {}

    // mkConstructor: () -> !ptr.void
    static FuncOp getOrInsertHpAlloc(PatternRewriter &rewriter,
                                     ModuleOp module) {
        const std::string name = "hpalloc_runtime";
        if (FuncOp fn = module.lookupSymbol<FuncOp>(name)) {
            return fn;
        }

        SmallVector<mlir::Type, 4> argTys = {};
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
        FuncOp fn =
            getOrInsertHpAlloc(rewriter, op->getParentOfType<ModuleOp>());

        llvm::errs() << "\nvvvvvHpAllocOp before:vvvvv:\n";
        fn.getParentOp()->dump();

        rewriter.setInsertionPointAfter(op);
        CallOp call = rewriter.create<CallOp>(op->getLoc(), fn);
        rewriter.replaceOp(op, call.getResults());

        llvm::errs() << "\n====after:====\n";
        fn.getParentOp()->dump();
        llvm::errs() << "\n^^^^^^\n";
        getchar();

        return success();
    }
};

class BadLoweringConversionPattern : public ConversionPattern {
   public:
    explicit BadLoweringConversionPattern(TypeConverter &tc,
                                          MLIRContext *context)
        : ConversionPattern(BadLoweringOp::getOperationName(), 1, tc, context) {
    }

    // badCall: (i1) -> ()
    static FuncOp getOrInsertBadCall(PatternRewriter &rewriter,
                                     ModuleOp module) {
        const std::string name = "badcall";
        if (FuncOp fn = module.lookupSymbol<FuncOp>(name)) {
            return fn;
        }

        SmallVector<mlir::Type, 4> argTys = {rewriter.getI1Type()};
        auto fntype = rewriter.getFunctionType(argTys, {});
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
        FuncOp fn =
            getOrInsertBadCall(rewriter, op->getParentOfType<ModuleOp>());

        llvm::errs() << "\nvvvvvBadLowering before:vvvvv\n";
        fn.getParentOp()->dump();

        rewriter.setInsertionPointAfter(op);
        // we call it with an incorrect argument; ergo, a bad std.call
        Value arg = rewriter.create<ConstantIntOp>(rewriter.getUnknownLoc(), 42,
                                                   rewriter.getI64Type());
        CallOp call = rewriter.create<CallOp>(op->getLoc(), fn, operands);
        rewriter.replaceOp(op, call.getResults());

        llvm::errs() << "\n====after:====\n";
        fn.getParentOp()->dump();
        llvm::errs() << "\n^^^^^\n";
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

        target.addDynamicallyLegalOp<FuncOp>([](FuncOp funcOp) {
            auto funcType = funcOp.getType();
            for (auto &arg : llvm::enumerate(funcType.getInputs())) {
                if (arg.value().isa<HiType>()) return false;
            }
            for (auto &arg : llvm::enumerate(funcType.getResults())) {
                if (arg.value().isa<HiType>()) return false;
            }
            return true;
        });

        target.addDynamicallyLegalOp<ReturnOp>([](ReturnOp ret) {
            auto retty = ret.getOperandTypes();
            for (Value arg : ret.getOperands()) {
                if (arg.getType().isa<HiType>()) return false;
            }
            return true;
        });

        HiTycon tycon(&getContext());


        ::llvm::DebugFlag = true;

        // applyPartialConversion | applyFullConversion
        mlir::OwningRewritePatternList patterns;
        patterns.insert<HpStoreConversionPattern>(tycon, &getContext());
        patterns.insert<HpAllocOpConversionPattern>(tycon, &getContext());
        patterns.insert<BadLoweringConversionPattern>(tycon, &getContext());
        patterns.insert<FuncOpLowering>(tycon, &getContext());
        patterns.insert<ReturnOpLowering>(tycon, &getContext());

        if (failed(mlir::applyFullConversion(getOperation(), target,
                                             std::move(patterns)))) {
            llvm::errs() << "\n===Hi lowering failed at Conversion===\n";
            getOperation()->print(llvm::errs());
            llvm::errs() << "\n===\n";
            signalPassFailure();
        };

        llvm::errs()
            << "\n===Module after conversion, before verification: ===\n";
        getOperation()->dump();

        llvm::errs() << "\n===Verifying lowering...===\n";
        ModuleOp mod = cast<ModuleOp>(getOperation());
        if (failed(mod.verify())) {
            llvm::errs() << "===Hi lowering failed at Verification===\n";
            getOperation()->print(llvm::errs());
            llvm::errs() << "\n===\n";
            signalPassFailure();
        }

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
