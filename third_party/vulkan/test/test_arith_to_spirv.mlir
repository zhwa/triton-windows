// Test: arith → SPIR-V conversion (basic float ops)
// RUN: python vulkan-opt.py %s --pipeline arith
// RUN: python vulkan-opt.py %s --pipeline arith --roundtrip

func.func @vector_add(%a: f32, %b: f32) -> f32 {
  %c = arith.addf %a, %b : f32
  return %c : f32
}

func.func @mul_add(%x: f32, %y: f32, %z: f32) -> f32 {
  %prod = arith.mulf %x, %y : f32
  %sum = arith.addf %prod, %z : f32
  return %sum : f32
}

func.func @int_ops(%a: i32, %b: i32) -> i32 {
  %sum = arith.addi %a, %b : i32
  %prod = arith.muli %sum, %b : i32
  return %prod : i32
}
