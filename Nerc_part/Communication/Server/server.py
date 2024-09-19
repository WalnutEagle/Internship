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
    server.add_insecure_port('[::]:8083') 
    server.start()
    print("Server running on port 8083")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
