import grpc
import time
import communication_pb2
import communication_pb2_grpc
import numpy as np

def log_time_info(step_name, start_time):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    elapsed_time = time.time() - start_time
    print(f"[{current_time}] {step_name} completed in {elapsed_time:.2f} seconds")

def send_tensor(stub, tensor):

    shape = tensor.shape
    data_type = "float32"  # You might want to make this dynamic based on the tensor's dtype
    tensor_data = tensor.tobytes()  # Convert tensor data to bytes

    tensor_request = communication_pb2.TensorRequest(shape=list(shape), data_type=data_type, data=tensor_data)
    start_time = time.time()
    response = stub.SendTensor(tensor_request)
    log_time_info("SendTensor", start_time)
    print(f"Client received tensor status from server: {response.status}")

def receive_tensor(stub):
    empty_request = communication_pb2.Empty()
    start_time = time.time()
    response = stub.ReceiveTensor(empty_request)
    log_time_info("ReceiveTensor", start_time)

    # Convert received data back to tensor
    shape = tuple(response.shape)  # Convert to tuple for reshaping
    tensor_data = np.frombuffer(response.data, dtype=np.float32).reshape(shape)  # Adjust dtype as needed
    print(f"Client received tensor status: {response.status}, shape: {response.shape}, data type: {response.data_type}")

    return tensor_data  # Return the tensor data

def run():
    start_time = time.time()
    
    # Establish channel and stub
    channel = grpc.insecure_channel('128.197.173.57:8083')  
    stub = communication_pb2_grpc.CommunicationServiceStub(channel)
    log_time_info("Channel and Stub Setup", start_time)

    # Example tensor to send (shape: 2x3)
    example_tensor = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    
    # Send and receive tensor
    send_tensor(stub, example_tensor)
    received_tensor = receive_tensor(stub)
    print(f"Received tensor data: {received_tensor}")

if __name__ == '__main__':
    run()
