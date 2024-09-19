'''
import grpc
import time
from concurrent import futures
import communication_pb2
import communication_pb2_grpc

class CommunicationService(communication_pb2_grpc.CommunicationServiceServicer):
    def log_time_info(self, step_name, start_time):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        elapsed_time = time.time() - start_time
        print(f"[{current_time}] {step_name} completed in {elapsed_time:.2f} seconds")

    # Functionality to receive tensor from client
    def SendTensor(self, request, context):
        start_time = time.time()
        print(f"Server received tensor with shape: {request.shape}, data type: {request.data_type}")
        # Process the tensor data as needed
        response = communication_pb2.TensorResponse(status="Tensor received successfully", 
                                                    shape=request.shape, 
                                                    data_type=request.data_type, 
                                                    data=request.data)  # Echo back the data for confirmation
        self.log_time_info("SendTensor", start_time)
        return response
    
    # Functionality to send tensor to client
    def ReceiveTensor(self, request, context):
        start_time = time.time()
        print("Server received request to send tensor")
        # Example tensor data (shape: 2x3, data type: float32)
        shape = [2, 3]
        data_type = "float32"
        tensor_data = bytes([0.7, 0.8, 0.9, 1.0, 1.1, 1.2])  # Example tensor data

        response = communication_pb2.TensorResponse(status="Tensor sent successfully",
                                                    shape=shape,
                                                    data_type=data_type,
                                                    data=tensor_data)
        self.log_time_info("ReceiveTensor", start_time)
        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    communication_pb2_grpc.add_CommunicationServiceServicer_to_server(CommunicationService(), server)
    server.add_insecure_port('[::]:8083')  # Communication port is 8083
    server.start()
    print("Server running on port 8083")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
'''


# import grpc
# import time
# from concurrent import futures
# import communication_pb2
# import communication_pb2_grpc
# import numpy as np

# class CommunicationService(communication_pb2_grpc.CommunicationServiceServicer):
#     def log_time_info(self, step_name, start_time):
#         current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#         elapsed_time = time.time() - start_time
#         print(f"[{current_time}] {step_name} completed in {elapsed_time:.2f} seconds")

#     def send_tensor(self, tensor):
#         # Convert tensor to appropriate format
#         shape = tensor.shape
#         data_type = "float32"  # or adjust based on your tensor type
#         tensor_data = tensor.tobytes()

#         tensor_request = communication_pb2.TensorRequest(shape=shape, data_type=data_type, data=tensor_data)

#         # Here you can simulate sending the tensor (e.g., through gRPC call)
#         # For simplicity, we'll just call SendTensor directly for testing
#         response = self.SendTensor(tensor_request, None)
#         return response  # You can modify this to return relevant data if needed

#     def receive_tensor(self):
#         # Simulate receiving a tensor (e.g., through gRPC call)
#         request = communication_pb2.Empty()  # Assume a simple request to trigger receiving
#         response = self.ReceiveTensor(request, None)

#         # Deserialize the received tensor data
#         tensor_shape = response.shape
#         tensor_data = np.frombuffer(response.data, dtype=np.float32).reshape(tensor_shape)
#         return tensor_data  # Return the tensor for assignment

# def serve():
#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
#     communication_pb2_grpc.add_CommunicationServiceServicer_to_server(CommunicationService(), server)
#     server.add_insecure_port('[::]:8083')  # Communication port is 8083
#     server.start()
#     print("Server running on port 8083")
#     CommunicationService.receive_tensor()
#     server.wait_for_termination()

# if __name__ == '__main__':
#     serve()


import grpc
import time
from concurrent import futures
import communication_pb2
import communication_pb2_grpc
import numpy as np

class CommunicationService(communication_pb2_grpc.CommunicationServiceServicer):
    def log_time_info(self, step_name, start_time):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        elapsed_time = time.time() - start_time
        print(f"[{current_time}] {step_name} completed in {elapsed_time:.2f} seconds")

    def SendTensor(self, request, context):
        start_time = time.time()
        print(f"Server received tensor with shape: {request.shape}, data type: {request.data_type}")

        # Echo back the data as confirmation
        response = communication_pb2.TensorResponse(
            status="Tensor received successfully",
            shape=request.shape,
            data_type=request.data_type,
            data=request.data
        )
        self.log_time_info("SendTensor", start_time)
        return response

    def ReceiveTensor(self, request, context):
        start_time = time.time()
        print("Server received request to send tensor")

        # Example tensor data (shape: 2x3, data type: float32)
        shape = [2, 3]
        data_type = "float32"
        
        tensor_data = np.array([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]], dtype=np.float32).tobytes()

        response = communication_pb2.TensorResponse(
            status="Tensor sent successfully",
            shape=shape,
            data_type=data_type,
            data=tensor_data
        )
        self.log_time_info("ReceiveTensor", start_time)
        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    communication_pb2_grpc.add_CommunicationServiceServicer_to_server(CommunicationService(), server)
    server.add_insecure_port('[::]:8083')  # Communication port is 8083
    server.start()
    print("Server running on port 8083")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
