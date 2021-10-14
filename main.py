from Model.Dexined import dexined_model

if "__name__" == "__main__":
    dexined_creation = dexined_model(input_shape=(320, 320, 3))
    my_model = dexined_creation.create_model()

    my_model.summary()
