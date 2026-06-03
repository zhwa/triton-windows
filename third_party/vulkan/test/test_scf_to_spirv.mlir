// Test: scf + arith → SPIR-V conversion (loops)
// RUN: python vulkan-opt.py %s --pipeline linalg

func.func @sum_loop(%n: index, %init: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f32
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> f32 {
    %new_acc = arith.addf %acc, %one : f32
    scf.yield %new_acc : f32
  }
  return %result : f32
}
