from Model.Dexined import dexined_model
from Run_Model import *

if __name__ == '__main__':
    dexined_creator = dexined_model(input_shape=(720, 1280, 3))
    my_model = dexined_creator.create_model()
    # my_model.summary()
    X_train, Y_train, X_test, Y_test = get_train_test_data()
    compile_and_fit_model(model=my_model, x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test)

