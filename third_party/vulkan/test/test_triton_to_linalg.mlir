// Test: triton-to-linalg pass — basic converter verification
// RUN: triton-opt --triton-to-linalg %s

// tt.splat, tt.make_range, tt.broadcast, tt.expand_dims
tt.func @test_tensor_ops(%scalar: f32) -> (tensor<128xf32>, tensor<128xi32>, tensor<128x1xf32>, tensor<128x64xf32>) {
  // tt.splat: scalar → tensor
  %splat = tt.splat %scalar : f32 -> tensor<128xf32>

  // tt.make_range: [0, 128)
  %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>

  // tt.expand_dims: 1D → 2D
  %expanded = tt.expand_dims %splat {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>

  // tt.broadcast: size-1 dim → full size
  %broadcast = tt.broadcast %expanded : tensor<128x1xf32> -> tensor<128x64xf32>

  tt.return %splat, %range, %expanded, %broadcast : tensor<128xf32>, tensor<128xi32>, tensor<128x1xf32>, tensor<128x64xf32>
}

// tt.get_program_id, tt.get_num_programs
tt.func @test_program_info() -> (i32, i32) {
  %pid = tt.get_program_id x : i32
  %npid = tt.get_num_programs x : i32
  tt.return %pid, %npid : i32, i32
}

// tt.dot: matrix multiply + accumulate
tt.func @test_matmul(%a: tensor<64x32xf32>, %b: tensor<32x64xf32>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %d = tt.dot %a, %b, %c : tensor<64x32xf32> * tensor<32x64xf32> -> tensor<64x64xf32>
  tt.return %d : tensor<64x64xf32>
}

// tt.trans: transpose
tt.func @test_transpose(%src: tensor<64x32xf32>) -> tensor<32x64xf32> {
  %t = tt.trans %src {order = array<i32: 1, 0>} : tensor<64x32xf32> -> tensor<32x64xf32>
  tt.return %t : tensor<32x64xf32>
}

// elementwise arith on tensors
tt.func @test_elementwise(%a: tensor<128xf32>, %b: tensor<128xf32>) -> tensor<128xf32> {
  %sum = arith.addf %a, %b : tensor<128xf32>
  %prod = arith.mulf %sum, %b : tensor<128xf32>
  tt.return %prod : tensor<128xf32>
}
