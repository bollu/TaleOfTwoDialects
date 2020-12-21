#pragma once
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "Dialects.h"

// conversion
// https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/examples/toy/Ch6/toyc.cpp
#include "mlir/Target/LLVMIR.h"

extern "C" {
const char *__asan_default_options() { return "detect_leaks=0"; }
}

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  // mlir::standalone::registerWorkerWrapperPass();
  // registerLzInterpretPass();
  mlir::DialectRegistry registry;

  registry.insert<mlir::HiDialect>();
  registry.insert<mlir::PtrDialect>();
  registry.insert<mlir::StandardOpsDialect>();
  registry.insert<mlir::AffineDialect>();
  return failed(mlir::MlirOptMain(argc, argv, "Standalone optimizer driver\n",registry, true));
}
