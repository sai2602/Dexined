from Model.Dexined import dexined_model
from Run_Model import predict

if __name__ == '__main__':
    dexined_creator = dexined_model()
    my_model = dexined_creator.create_model()
    predict(my_model)
