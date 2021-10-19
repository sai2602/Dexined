from Model.Dexined import dexined_model
from Run_Model import *
from utls.helpers import *


if __name__ == '__main__':
    args = configuration()
    dexined_creator = dexined_model(input_shape=(720, 1280, 3))
    my_model = dexined_creator.create_model()
    my_model = compile_model(my_model)
    X_train, Y_train, X_test, Y_test = get_train_test_data(args.data_dir)
    if args.model_state == 'train':
        train_model(my_model, X_train, Y_train, X_test, Y_test, batch_size=args.batch_size, epochs=args.epochs,
                    save_dir=args.save_dir)
