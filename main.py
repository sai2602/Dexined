from Model.Dexined import dexined_model

dexined_creator = dexined_model()
my_model = dexined_creator.create_model()
my_model.summary()
