#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// cmake --build . -j 4
// Usage:
// ./minimal /home/ejfdelgado/desarrollo/vaale/tensor_python/models/petals.tflite 5.0 3.2 1.2 0.2
// ./minimal /home/ejfdelgado/desarrollo/vaale/tensor_python/models/petals.tflite 5.9 3.2 4.8 1.8
// ./minimal /home/ejfdelgado/desarrollo/vaale/tensor_python/models/petals.tflite 7.2 3.6 6.1 2.5

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 6) {
    fprintf(stderr, "minimal <tflite model> <v0> <v1> <v2> <v3>\n");
    return 1;
  }
  const char* filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`
  // 5.0, 3.2, 1.2, 0.2
  
  interpreter->typed_input_tensor<float>(0)[0] = atof(argv[2]);
  interpreter->typed_input_tensor<float>(0)[1] = atof(argv[3]);
  interpreter->typed_input_tensor<float>(0)[2] = atof(argv[4]);
  interpreter->typed_input_tensor<float>(0)[3] = atof(argv[5]);
  

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("\n\n=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
  float* output = interpreter->typed_output_tensor<float>(0);
  printf("0? %.6f\n", output[0]);
  printf("1? %.6f\n", output[1]);
  printf("2? %.6f\n", output[2]);

  return 0;
}
