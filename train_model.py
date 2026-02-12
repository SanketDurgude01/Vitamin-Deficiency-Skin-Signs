import model
train_gen, val_gen = model.create_data_generators(
       'data/processed/train',
       'data/processed/validation'
   )
history = model.train(train_gen, val_gen, epochs=50)
model.save_model(trained_model, "vitamin_deficiency_model_final.h5")
