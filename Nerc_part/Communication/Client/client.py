import grpc
import time
import communication_pb2
import communication_pb2_grpc

def log_time_info(step_name, start_time):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    elapsed_time = time.time() - start_time
    print(f"[{current_time}] {step_name} completed in {elapsed_time:.2f} seconds")

def send_tensor(stub):
    # Example tensor data (shape: 2x3, data type: float32)
    shape = [2, 3]
    data_type = "float32"
    tensor_data = bytearray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # Replace with actual tensor data

    tensor_request = communication_pb2.TensorRequest(shape=shape, data_type=data_type, data=tensor_data)
    start_time = time.time()
    response = stub.SendTensor(tensor_request)
    log_time_info("SendTensor", start_time)
    print(f"Client received tensor status from server: {response.status}")

def receive_tensor(stub):
    empty_request = communication_pb2.Empty()
    start_time = time.time()
    response = stub.ReceiveTensor(empty_request)
    log_time_info("ReceiveTensor", start_time)
    print(f"Client received tensor status: {response.status}, shape: {response.shape}, data type: {response.data_type}")

def run():
    start_time = time.time()
    
    # Establish channel and stub
    channel = grpc.insecure_channel('128.197.164.40:8083')
    stub = communication_pb2_grpc.CommunicationServiceStub(channel)
    log_time_info("Channel and Stub Setup", start_time)

    # Send and receive tensor
    send_tensor(stub)
    receive_tensor(stub)

if __name__ == '__main__':
    run()
