# A tale of two dialects

How should one use the `typeConverter`?

##### Problem statement: `addTargetMaterialization` is broken.

##### Step 1: use the API as it is meant to be used.


```cpp
class HiTycon : public mlir::TypeConverter {
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
} // end hitycon
```

```cpp
class HpAllocOpConversionPattern : public ConversionPattern {
LogicalResult matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const override {
    FuncOp fn =
        getOrInsertHpAlloc(rewriter, op->getParentOfType<ModuleOp>());

    rewriter.setInsertionPointAfter(op);
    rewriter.replaceOpWithNewOp<CallOp>(op, fn);

    llvm::errs() << "\n====\n";
    fn.getParentOp()->dump();
    llvm::errs() << "\n===\n";
    getchar();

    return success();
}
} // end HpAllocConversion Pattern

```

- We run this on the module:

```cpp
module {
    func @main() -> !hi.hpnode {
        %x = hi.hpAlloc
        return %x : !hi.hpnode
    }
}
```

- run with `./mir-opt --hi-lower`


```
Legalizing operation : 'hi.hpAlloc'(0x607000003350) {
  %0 = "hi.hpAlloc"() : () -> !hi.hpnode

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'hi.hpAlloc -> ()' {
    ** Insert  : 'func'(0x608000002220)
    ** Insert  : 'std.call'(0x6080000022a0)
    ** Replace : 'hi.hpAlloc'(0x607000003350)

====
module  {
  func private @hpalloc_runtime() -> !ptr.void
  func @main() -> !hi.hpnode {
    %0 = "hi.hpAlloc"() : () -> !hi.hpnode
    %1 = call @hpalloc_runtime() : () -> !ptr.void
    return %0 : !hi.hpnode <= wut?
  }
}

===


    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x6080000021a0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'std.call'(0x608000002220) {
      %1 = "std.call"() {callee = @hpalloc_runtime} : () -> !ptr.void

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'std.return'(0x60c0000016c0) {
  "std.return"(%0) : (!hi.hpnode) -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'module_terminator'(0x607000003430) {
  "module_terminator"() : () -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//
return-heapnode.mlir:8:14: error: failed to materialize conversion for result #0 of operation 'hi.hpAlloc' that remained live after conversion
        %x = hi.hpAlloc <- wut?
             ^
return-heapnode.mlir:8:14: note: see current operation: %0 = "hi.hpAlloc"() : () -> !hi.hpnode
return-heapnode.mlir:9:9: note: see existing live user here: return %0 : !hi.hpnode
        return %x : !hi.hpnode
```

- So this doesn't really seem to be doing the right thing??
- Then you read the API to see the nugget:

```cpp
  /// Replaces the result op with a new op that is created without verification.
  /// The result values of the two ops must be the same types.
  template <typename OpTy, typename... Args>
  void replaceOpWithNewOp(Operation *op, Args &&... args) {
    auto newOp = create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
    replaceOpWithResultsOfAnotherOp(op, newOp.getOperation());
  }
```


- `The result values of the two ops must be the same types.` :(



Am I supposed to call the conversion directly? Seems unlikely:

```cpp
[I] /home/bollu/work/mlir/llvm-project/mlir > ag materializeTargetConversion
lib/Transforms/Utils/DialectConversion.cpp
1073:      newOperand = converter->materializeTargetConversion(

include/mlir/Transforms/DialectConversion.h
216:  Value materializeTargetConversion(OpBuilder &builder, Location loc,
[I] /home/bollu/work/mlir/llvm-project/mlir >
```
