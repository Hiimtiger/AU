import os

os.makedirs( "Train_Model/TRAINING_IMAGES", exist_ok = True )
os.makedirs( "Train_Model/TRAINING_MASKS", exist_ok = True )

os.makedirs( "USE_MODEL/INPUT_IMAGES", exist_ok = True )
os.makedirs( "USE_MODEL/OUTPUT_MASKS", exist_ok = True )
os.makedirs( "USE_MODEL/saved_models", exist_ok = True )

os.makedirs( "FINETUNE_MODEL/INPUT_IMAGES", exist_ok = True )
os.makedirs( "FINETUNE_MODEL/INPUT_MASKS", exist_ok = True )

os.makedirs( "utils/Sample_Images", exist_ok = True )
os.makedirs( "utils/training_images", exist_ok = True )
os.makedirs( "utils/training_masks", exist_ok = True )