import onnxruntime as ort
import numpy as np

def test_onnx_model():
    print("Loading ONNX model...")
    session = ort.InferenceSession("encoder.onnx", providers=['CPUExecutionProvider'])
    
    # Get input details
    input_details = session.get_inputs()
    output_details = session.get_outputs()
    
    print("\nExpected Inputs:")
    for i in input_details:
        print(f" - Name: {i.name}, Shape: {i.shape}, Type: {i.type}")
        
    print("\nExpected Outputs:")
    for o in output_details:
        print(f" - Name: {o.name}, Shape: {o.shape}, Type: {o.type}")
        
    # Create dummy numpy inputs
    batch_size = 1
    seq_length = 16
    
    # input_ids and attention_mask as int64
    dummy_input_ids = np.zeros((batch_size, seq_length), dtype=np.int64)
    dummy_attention_mask = np.ones((batch_size, seq_length), dtype=np.int64)
    
    print("\nRunning inference...")
    inputs = {
        "input_ids": dummy_input_ids,
        "attention_mask": dummy_attention_mask
    }
    
    outputs = session.run(None, inputs)
    
    print("\nInference successful! Output shape:")
    print(outputs[0].shape)
    
    if outputs[0].shape == (batch_size, seq_length, 768):
        print("\n✅ VALIDATION PASSED: The shape (batch, seq_len, 768) matches DeBERTa base hidden size!")
    else:
        print("\n❌ VALIDATION FAILED: Unexpected shape.")

if __name__ == "__main__":
    test_onnx_model()
